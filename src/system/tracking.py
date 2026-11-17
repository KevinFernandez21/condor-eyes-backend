"""Tracker mínimo por IoU y asignación de zona, para el rol ``tracker``.

No pretende sustituir a NvDCF/ByteTrack (ARCH-003): da ``track_ref`` estables
y una zona por posición horizontal para que la fusión y el dashboard tengan
datos reales en laptop. Solo trabaja con metadata (cajas y números).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from agents.handlers import TrackFn
from bus import MetadataEnvelope

from .config import ZoneConfig

PERSON_CLASS = 0


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _finite_box(raw: Any) -> list[float] | None:
    try:
        box = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    return box


class SimpleTracker:
    """Seguimiento por IoU (misma clase) con envejecimiento; ids por stream.

    Si la detección ya trae ``track_id`` (fuente pre-rastreada, como las escenas
    sintéticas) se respeta y el ``track_ref`` es ``<stream>/<track_id>``; el
    resto se asocia por IoU. Sigue cualquier clase (personas, vehículos y
    movimiento sin clase); la fusión filtra las personas.
    """

    def __init__(self, *, iou_min: float = 0.2, max_age: int = 10) -> None:
        self._iou_min = iou_min
        self._max_age = max_age
        # stream_id -> {numero: [caja, edad sin ver, clase]}
        self._tracks: dict[str, dict[int, list[Any]]] = {}
        self._next: dict[str, int] = {}

    def update(
        self, stream_id: str, detections: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        tracks = self._tracks.setdefault(stream_id, {})
        out: list[dict[str, Any]] = []
        boxes: list[tuple[list[float], float, int, Any]] = []
        for det in detections:
            box = _finite_box(det.get("xyxy"))
            cls = det.get("cls")
            if box is None or isinstance(cls, bool) or not isinstance(cls, int):
                continue
            conf = float(det.get("conf", 0.0))
            external = det.get("track_id")
            if isinstance(external, int) and not isinstance(external, bool):
                out.append(self._track(stream_id, external, box, cls, conf, det))
            else:
                boxes.append((box, conf, cls, det))
        pairs = sorted(
            (
                (_iou(info[0], box), number, index)
                for number, info in tracks.items()
                for index, (box, _, cls, _) in enumerate(boxes)
                if info[2] == cls
            ),
            reverse=True,
        )
        assigned: dict[int, int] = {}
        used: set[int] = set()
        for score, number, index in pairs:
            if score < self._iou_min:
                break
            if number in assigned or index in used:
                continue
            assigned[number] = index
            used.add(index)
        for number in list(tracks):
            if number in assigned:
                tracks[number][0] = boxes[assigned[number]][0]
                tracks[number][1] = 0
            else:
                tracks[number][1] += 1
                if tracks[number][1] > self._max_age:
                    del tracks[number]
        for index, (box, conf, cls, det) in enumerate(boxes):
            found = next((n for n, i in assigned.items() if i == index), None)
            if found is None:
                number = self._next.get(stream_id, 0) + 1
                self._next[stream_id] = number
                tracks[number] = [box, 0, cls]
            else:
                number = found
            out.append(self._track(stream_id, number, box, cls, conf, det))
        return out

    @staticmethod
    def _track(
        stream_id: str,
        number: int,
        box: list[float],
        cls: int,
        conf: float,
        det: Mapping[str, Any],
    ) -> dict[str, Any]:
        track: dict[str, Any] = {
            "track_ref": f"{stream_id}/{number}",
            "track_id": number,
            "xyxy": box,
            "cls": cls,
            "conf": conf,
        }
        if isinstance(det.get("label"), str):
            track["label"] = det["label"]
        return track


def zone_for(zones: Sequence[ZoneConfig], center_x: float) -> ZoneConfig | None:
    """Zona cuya franja horizontal contiene el centro (fracción de 0 a 1)."""
    for zone in zones:
        if zone.x_min <= center_x < zone.x_max:
            return zone
    return zones[-1] if zones and center_x >= 1.0 else None


def make_track_fn(
    zones: Sequence[ZoneConfig],
    by_stream: Mapping[str, Sequence[ZoneConfig]] | None = None,
) -> TrackFn:
    """``TrackFn`` con estado: detecciones de un frame -> tracks con zona.

    ``by_stream`` da zonas propias por cámara; ``zones`` es el valor por defecto.
    """
    tracker = SimpleTracker()

    def track_fn(envelope: MetadataEnvelope) -> Mapping[str, Any]:
        payload = envelope.payload
        stream_id = str(envelope.stream_id or payload.get("stream_id") or "stream")
        width = float(payload.get("width") or 0) or 1.0
        tracks = tracker.update(stream_id, list(payload.get("detections", [])))
        stream_zones = by_stream.get(stream_id, zones) if by_stream else zones
        for track in tracks:
            x0, _, x1, _ = track["xyxy"]
            zone = zone_for(stream_zones, ((x0 + x1) / 2.0) / width)
            track["zone_id"] = zone.zone_id if zone else None
            track["stream_id"] = stream_id
        return {
            "stream_id": stream_id,
            "frame_index": payload.get("frame_index"),
            "timestamp": payload.get("timestamp"),
            "tracks": tracks,
        }

    return track_fn


class ZoneEntryRule:
    """Regla del rol ``event``: emite un evento cuando un track cambia de zona."""

    def __init__(self, zones: Sequence[ZoneConfig], *, capacity: int = 1000) -> None:
        self._restricted = {z.zone_id: z.restricted for z in zones}
        self._last: dict[str, str | None] = {}
        self._capacity = capacity

    def __call__(self, envelope: MetadataEnvelope) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for track in envelope.payload.get("tracks", []):
            ref, zone = track.get("track_ref"), track.get("zone_id")
            if track.get("cls") == -1:
                continue  # movimiento sin clase: no es un evento de zona
            if not isinstance(ref, str) or self._last.get(ref, "") == zone:
                continue
            previous = self._last.get(ref)
            self._last[ref] = zone
            events.append(
                {
                    "kind": "track.zone_entered",
                    "track_ref": ref,
                    "zone_id": zone,
                    "label": track.get("label"),
                    "from_zone": previous,
                    "restricted": bool(self._restricted.get(zone or "", False)),
                    "stream_id": envelope.stream_id,
                }
            )
        while len(self._last) > self._capacity:
            self._last.pop(next(iter(self._last)))
        return events
