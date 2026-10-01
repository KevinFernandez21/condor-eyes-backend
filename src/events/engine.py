"""Baseline de reglas: intrusión, merodeo y aglomeración sobre tracks.

El motor consume observaciones en orden temporal por cámara y devuelve eventos.
La decisión de merodeo es intercambiable (`LoiterDecider`) para comparar la regla
con un modelo aprendido usando exactamente el mismo estado y la misma salida.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Protocol

from .model import (
    AuthorizedPolicy,
    Event,
    EventPolicy,
    EventType,
    Severity,
    TrackObservation,
    Zone,
)


def point_in_polygon(
    x: float, y: float, polygon: Sequence[tuple[float, float]]
) -> bool:
    """Ray casting; los puntos sobre el borde pueden caer de cualquier lado."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


@dataclass(frozen=True, slots=True)
class LoiterFeatures:
    """Estado de un track dentro de una zona de merodeo en un instante."""

    dwell_s: float
    window_s: float
    net_disp: float
    path_len: float
    # (stream_id, track_id, zona): permite a un decisor con estado suavizar en el tiempo.
    key: tuple[str, int, str] = ("", -1, "")

    @property
    def straightness(self) -> float:
        return self.net_disp / self.path_len if self.path_len > 1e-9 else 0.0


class LoiterDecider(Protocol):
    def __call__(self, f: LoiterFeatures, zone: Zone) -> tuple[bool, float]:
        """Devuelve (es_merodeo, confianza)."""
        ...


def rule_loiter(f: LoiterFeatures, zone: Zone) -> tuple[bool, float]:
    """Merodeo = permanencia >= umbral y poco desplazamiento neto en esa ventana."""
    assert zone.loiter_dwell_s is not None
    if f.dwell_s < zone.loiter_dwell_s or f.net_disp > zone.loiter_max_disp:
        return False, 0.0
    return True, 0.5 + 0.5 * (1.0 - f.net_disp / zone.loiter_max_disp)


@dataclass
class _ZoneState:
    entered_t: float | None = None
    last_inside_t: float | None = None
    run_start_t: float | None = None
    fired: set[EventType] = field(default_factory=set)


@dataclass
class _TrackState:
    last_t: float
    authorized: bool
    history: deque[tuple[float, float, float]]
    zones: dict[str, _ZoneState] = field(default_factory=dict)


@dataclass
class _CrowdState:
    since_t: float | None = None
    fired: bool = False


class EventEngine:
    def __init__(
        self,
        zones: Iterable[Zone],
        policy: EventPolicy | None = None,
        loiter_decider: LoiterDecider = rule_loiter,
    ) -> None:
        self.zones = tuple(zones)
        self.policy = policy or EventPolicy()
        self.loiter_decider = loiter_decider
        self._by_stream: dict[str, list[Zone]] = {}
        for z in self.zones:
            self._by_stream.setdefault(z.stream_id, []).append(z)
        dwell = [z.loiter_dwell_s for z in self.zones if z.loiter_dwell_s]
        # Historial suficiente para la ventana de merodeo más larga.
        self._history_s = max(dwell, default=0.0) + self.policy.max_gap_s
        self._tracks: dict[tuple[str, int], _TrackState] = {}
        self._crowds: dict[str, _CrowdState] = {
            z.name: _CrowdState() for z in self.zones
        }

    def reset(self) -> None:
        self._tracks.clear()
        self._crowds = {z.name: _CrowdState() for z in self.zones}

    def process(self, observations: Iterable[TrackObservation]) -> list[Event]:
        events: list[Event] = []
        for obs in observations:
            events.extend(self.update(obs))
        return events

    def update(self, obs: TrackObservation) -> list[Event]:
        zones = self._by_stream.get(obs.stream_id, [])
        self._expire(obs.stream_id, obs.t)
        key = (obs.stream_id, obs.track_id)
        st = self._tracks.get(key)
        if st is None or obs.t - st.last_t > self.policy.max_gap_s or obs.t < st.last_t:
            st = _TrackState(obs.t, obs.authorized, deque())
            self._tracks[key] = st
        st.last_t = obs.t
        st.authorized = obs.authorized
        st.history.append((obs.t, obs.x, obs.y))
        while st.history and obs.t - st.history[0][0] > self._history_s:
            st.history.popleft()

        events: list[Event] = []
        for zone in zones:
            zs = st.zones.setdefault(zone.name, _ZoneState())
            if point_in_polygon(obs.x, obs.y, zone.polygon):
                if zs.entered_t is None:
                    zs.entered_t = obs.t
                if zs.run_start_t is None:
                    zs.run_start_t = obs.t
                zs.last_inside_t = obs.t
            else:
                zs.run_start_t = None
                if (
                    zs.last_inside_t is not None
                    and obs.t - zs.last_inside_t >= self.policy.exit_grace_s
                ):
                    st.zones[zone.name] = zs = _ZoneState()
            events.extend(self._track_rules(obs, st, zone, zs))
            if zone.crowd_threshold is not None:
                events.extend(self._crowd_rule(obs.t, zone))
        return events

    def _expire(self, stream_id: str, t: float) -> None:
        stale = [
            k
            for k, s in self._tracks.items()
            if k[0] == stream_id and t - s.last_t > self.policy.max_gap_s
        ]
        for k in stale:
            del self._tracks[k]

    def _severity(self, authorized: bool) -> Severity:
        if not authorized:
            return Severity.ALERT
        if self.policy.authorized is AuthorizedPolicy.SUPPRESS:
            return Severity.SUPPRESSED
        return Severity.REVIEW

    def _track_rules(
        self, obs: TrackObservation, st: _TrackState, zone: Zone, zs: _ZoneState
    ) -> list[Event]:
        out: list[Event] = []
        if zs.entered_t is None:
            return out
        if (
            zone.restricted
            and EventType.INTRUSION not in zs.fired
            and zs.run_start_t is not None
            and obs.t - zs.run_start_t >= zone.intrusion_min_s
        ):
            zs.fired.add(EventType.INTRUSION)
            inside = [h for h in st.history if h[0] >= zs.run_start_t]
            out.append(
                Event(
                    type=EventType.INTRUSION,
                    stream_id=obs.stream_id,
                    zone=zone.name,
                    track_ids=(obs.track_id,),
                    start_t=zs.run_start_t,
                    detect_t=obs.t,
                    confidence=1.0,
                    severity=self._severity(st.authorized),
                    authorized=st.authorized,
                    evidence={
                        "inside_s": round(obs.t - zs.run_start_t, 3),
                        "observations_inside": len(inside),
                        "entry_point": [round(inside[0][1], 4), round(inside[0][2], 4)],
                        "last_point": [round(obs.x, 4), round(obs.y, 4)],
                    },
                )
            )
        if zone.loiter_dwell_s is not None and EventType.LOITERING not in zs.fired:
            f = self._loiter_features(obs.t, st, zs.entered_t, zone.loiter_dwell_s)
            f = replace(f, key=(obs.stream_id, obs.track_id, zone.name))
            ok, conf = self.loiter_decider(f, zone)
            if ok:
                zs.fired.add(EventType.LOITERING)
                out.append(
                    Event(
                        type=EventType.LOITERING,
                        stream_id=obs.stream_id,
                        zone=zone.name,
                        track_ids=(obs.track_id,),
                        start_t=zs.entered_t,
                        detect_t=obs.t,
                        confidence=conf,
                        severity=self._severity(st.authorized),
                        authorized=st.authorized,
                        evidence={
                            "dwell_s": round(f.dwell_s, 3),
                            "window_s": round(f.window_s, 3),
                            "net_displacement": round(f.net_disp, 4),
                            "path_length": round(f.path_len, 4),
                            "last_point": [round(obs.x, 4), round(obs.y, 4)],
                        },
                    )
                )
        return out

    @staticmethod
    def _loiter_features(
        t: float, st: _TrackState, entered_t: float, window: float
    ) -> LoiterFeatures:
        start = max(entered_t, t - window)
        pts = [h for h in st.history if h[0] >= start]
        path = sum(math.dist(a[1:], b[1:]) for a, b in pairwise(pts))
        disp = math.dist(pts[0][1:], pts[-1][1:]) if pts else 0.0
        return LoiterFeatures(
            dwell_s=t - entered_t, window_s=t - start, net_disp=disp, path_len=path
        )

    def _crowd_rule(self, t: float, zone: Zone) -> list[Event]:
        assert zone.crowd_threshold is not None
        members = sorted(
            (k[1], s.authorized)
            for k, s in self._tracks.items()
            if k[0] == zone.stream_id
            and (zs := s.zones.get(zone.name)) is not None
            and zs.entered_t is not None
        )
        cs = self._crowds[zone.name]
        if len(members) < zone.crowd_threshold:
            cs.since_t, cs.fired = None, False
            return []
        if cs.since_t is None:
            cs.since_t = t
        if cs.fired or t - cs.since_t < zone.crowd_dwell_s:
            return []
        cs.fired = True
        all_auth = all(a for _, a in members)
        return [
            Event(
                type=EventType.CROWDING,
                stream_id=zone.stream_id,
                zone=zone.name,
                track_ids=tuple(i for i, _ in members),
                start_t=cs.since_t,
                detect_t=t,
                confidence=min(1.0, len(members) / (zone.crowd_threshold + 1)),
                severity=self._severity(all_auth),
                authorized=all_auth,
                evidence={
                    "count": len(members),
                    "sustained_s": round(t - cs.since_t, 3),
                },
            )
        ]
