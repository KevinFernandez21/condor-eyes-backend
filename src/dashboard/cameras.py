"""Vista multicámara: metadata de cámaras y detecciones listas para dibujar.

No hay frames: cada cámara se pinta como una escena sintética y las cajas salen
de la metadata de detección (normalizadas a [0, 1] respecto del tamaño del
frame, que por defecto se asume 1920x1080).
"""

from __future__ import annotations

import math
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

DEFAULT_FRAME = (1920, 1080)
MAX_DETECTIONS = 100
CLASSES = ("person", "vehicle", "motion")
CLASS_LABEL = {"person": "Persona", "vehicle": "Vehículo", "motion": "Movimiento"}
SCENES = ("lobby", "parking", "perimeter", "warehouse", "corridor")

_VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle", "van"}
_COCO_VEHICLES = {1, 2, 3, 5, 7}  # bicicleta, auto, moto, bus, camión
_SCENE_KEYWORDS = (
    ("lobby", ("entrada", "lobby", "puerta", "acceso", "recepci", "hall")),
    ("parking", ("estacion", "parking", "parqueo", "garaje", "vehic")),
    ("perimeter", ("perim", "perí", "cerca", "fence", "exterior", "patio")),
    ("warehouse", ("bodega", "vault", "almac", "depósito", "deposito", "sala")),
    ("corridor", ("pasillo", "corredor", "corridor")),
)
_ONLINE = {"connected", "up", "running", "ok", "online", "simulated"}
_DEGRADED = {"reconnecting", "degraded", "connecting", "starting", "retrying"}
_OFFLINE = {"failed", "stopped", "down", "offline", "idle", "error"}
STATE_LABEL = {"online": "En línea", "degraded": "Degradada", "offline": "Sin señal"}


def detection_class(det: Mapping[str, Any]) -> str:
    """person | vehicle | motion; la etiqueta manda sobre el id de clase."""
    label = det.get("label")
    if isinstance(label, str) and label:
        name = label.lower()
        if name == "person":
            return "person"
        return "vehicle" if name in _VEHICLE_LABELS else "motion"
    cls = det.get("cls")
    if isinstance(cls, int) and not isinstance(cls, bool):
        if cls == 0:
            return "person"
        if cls in _COCO_VEHICLES:
            return "vehicle"
    return "motion"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def normalize_detections(
    detections: Sequence[Any], width: float, height: float
) -> list[dict[str, Any]]:
    """Cajas en [0, 1]; descarta lo malformado y acota la cantidad."""
    out: list[dict[str, Any]] = []
    if not isinstance(detections, (list, tuple)) or width <= 0 or height <= 0:
        return out
    for det in detections:
        if len(out) >= MAX_DETECTIONS:
            break
        if not isinstance(det, Mapping):
            continue
        raw = det.get("xyxy")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        values = [_num(v) for v in raw]
        if any(v is None for v in values):
            continue
        x0, y0, x1, y1 = (float(v) for v in values if v is not None)
        box = [
            round(min(max(x0 / width, 0.0), 1.0), 4),
            round(min(max(y0 / height, 0.0), 1.0), 4),
            round(min(max(x1 / width, 0.0), 1.0), 4),
            round(min(max(y1 / height, 0.0), 1.0), 4),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        cls = detection_class(det)
        track = det.get("track_id")
        out.append(
            {
                "box": box,
                "cls": cls,
                "label": CLASS_LABEL[cls],
                "conf": _num(det.get("conf")),
                "track_id": track if isinstance(track, (int, str)) and not isinstance(track, bool) else None,
            }
        )
    return out


def scene_kind(scene: str | None, camera_id: str) -> str:
    """Tipo de escena sintética: por palabra clave o, si no, estable por id."""
    text = (scene or "").lower()
    for kind, words in _SCENE_KEYWORDS:
        if any(w in text for w in words):
            return kind
    return SCENES[zlib.crc32(camera_id.encode("utf-8")) % len(SCENES)]


def camera_state(raw: Any) -> str | None:
    """online | degraded | offline según el estado que reporte el stream."""
    if not isinstance(raw, str):
        return None
    name = raw.lower().rsplit(".", 1)[-1]
    if name in _ONLINE:
        return "online"
    if name in _DEGRADED:
        return "degraded"
    if name in _OFFLINE:
        return "offline"
    return None


def _text(value: Any) -> str | None:
    return value.strip()[:80] if isinstance(value, str) and value.strip() else None


class _Camera:
    __slots__ = (
        "dets",
        "dets_at",
        "error",
        "frame",
        "id",
        "name",
        "reported",
        "reported_at",
        "scene",
        "seen_at",
        "tracks",
        "tracks_at",
    )  # fmt: skip

    def __init__(self, camera_id: str) -> None:
        self.id = camera_id
        self.name: str | None = None
        self.scene: str | None = None
        self.reported: str | None = None
        self.reported_at = 0.0
        self.error: str | None = None
        self.frame = DEFAULT_FRAME
        self.dets: list[dict[str, Any]] = []
        self.dets_at = 0.0
        self.tracks: list[dict[str, Any]] = []
        self.tracks_at = 0.0
        self.seen_at = 0.0


class CameraRegistry:
    """Última metadata conocida por cámara; acotado en número de cámaras."""

    TRACKS_FRESH_S = 3.0

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        stale_s: float = 8.0,
        max_cameras: int = 64,
    ) -> None:
        self._clock = clock
        self._stale_s = stale_s
        self._max = max_cameras
        self._cams: OrderedDict[str, _Camera] = OrderedDict()

    def _get(self, camera_id: str) -> _Camera | None:
        cam = self._cams.get(camera_id)
        if cam is None:
            if len(self._cams) >= self._max:
                return None
            cam = self._cams[camera_id] = _Camera(camera_id)
        return cam

    def observe(self, envelope: Mapping[str, Any]) -> None:
        if not isinstance(envelope, Mapping):
            return
        topic = envelope.get("topic")
        payload = envelope.get("payload")
        if topic not in ("stream.status", "system.health", "vision.detections", "vision.tracks"):
            return
        if not isinstance(payload, Mapping):
            return
        camera_id = envelope.get("stream_id") or payload.get("stream_id")
        if not isinstance(camera_id, str) or not camera_id:
            return
        cam = self._get(camera_id)
        if cam is None:
            return
        now = self._clock()
        cam.seen_at = now
        nested = payload.get("camera")
        meta: Mapping[str, Any] = nested if isinstance(nested, Mapping) else payload
        cam.name = _text(meta.get("name") or meta.get("camera_name")) or cam.name
        cam.scene = _text(meta.get("scene")) or cam.scene
        width, height = _num(payload.get("width")), _num(payload.get("height"))
        if width and height and width > 0 and height > 0:
            cam.frame = (int(width), int(height))
        if topic in ("stream.status", "system.health"):
            state = camera_state(payload.get("state"))
            if state is not None:
                cam.reported, cam.reported_at = state, now
                cam.error = _text(payload.get("last_error"))
        elif topic == "vision.detections":
            cam.dets = normalize_detections(payload.get("detections") or [], *cam.frame)
            cam.dets_at = now
        else:
            cam.tracks = normalize_detections(payload.get("tracks") or [], *cam.frame)
            cam.tracks_at = now

    def views(self) -> list[dict[str, Any]]:
        now = self._clock()
        out: list[dict[str, Any]] = []
        for cam in sorted(self._cams.values(), key=lambda c: c.id):
            stale = now - cam.seen_at > self._stale_s
            if stale:
                state, message = "offline", "Sin señal: no llegan mensajes de esta cámara"
            elif cam.reported in ("offline", "degraded"):
                state = cam.reported
                message = cam.error or ("Cámara caída" if state == "offline" else "Cámara degradada")
            else:
                state, message = "online", None
            use_tracks = cam.tracks and now - cam.tracks_at <= self.TRACKS_FRESH_S
            dets = [] if state == "offline" else (cam.tracks if use_tracks else cam.dets)
            counts = {c: 0 for c in CLASSES}
            for det in dets:
                counts[det["cls"]] += 1
            out.append(
                {
                    "id": cam.id,
                    "name": cam.name or cam.id,
                    "scene": cam.scene,
                    "scene_kind": scene_kind(cam.scene, cam.id),
                    "state": state,
                    "state_label": STATE_LABEL[state],
                    "message": message,
                    "frame": {"w": cam.frame[0], "h": cam.frame[1]},
                    "detections": dets,
                    "counts": counts,
                    "last_seen_age_s": round(now - cam.seen_at, 1),
                }
            )
        return out
