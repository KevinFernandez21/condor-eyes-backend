"""Tipos del detector de eventos: observaciones de track, zonas y eventos.

Las coordenadas son el punto de apoyo (centro inferior de la caja) normalizado
a [0, 1] respecto al frame, para que las zonas no dependan de la resolución.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from bus.hub import MetadataEnvelope, Topic


class EventType(StrEnum):
    LOITERING = "loitering"
    INTRUSION = "zone_intrusion"
    CROWDING = "crowding"


class Severity(StrEnum):
    ALERT = "alert"
    # Evento de persona autorizada: se rebaja a revisión, la evidencia se conserva.
    REVIEW = "review"
    # Política `suppress`: no alerta, pero el evento y su evidencia se publican.
    SUPPRESSED = "suppressed"


class AuthorizedPolicy(StrEnum):
    DOWNGRADE = "downgrade"
    SUPPRESS = "suppress"


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """Una posición de un track local de una cámara (salida del tracker)."""

    stream_id: str
    track_id: int
    t: float
    x: float
    y: float
    # Lo aporta el agente de identidad/fusión; el detector de eventos no decide.
    authorized: bool = False


@dataclass(frozen=True, slots=True)
class Zone:
    """Polígono de una cámara con las reglas que aplican dentro de él."""

    name: str
    stream_id: str
    polygon: tuple[tuple[float, float], ...]
    restricted: bool = False
    # Tiempo continuo dentro de la zona a partir del cual se considera intrusión.
    intrusion_min_s: float = 0.6
    # Merodeo: permanencia mínima y desplazamiento neto máximo en esa ventana.
    loiter_dwell_s: float | None = None
    loiter_max_disp: float = 0.12
    # Aglomeración: tracks simultáneos y tiempo sostenido.
    crowd_threshold: int | None = None
    crowd_dwell_s: float = 5.0

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"La zona {self.name} necesita al menos 3 vértices")
        if self.loiter_dwell_s is not None and self.loiter_dwell_s <= 0:
            raise ValueError("loiter_dwell_s debe ser positivo")
        if self.crowd_threshold is not None and self.crowd_threshold < 2:
            raise ValueError("crowd_threshold debe ser >= 2")


@dataclass(frozen=True, slots=True)
class EventPolicy:
    # Un hueco mayor (oclusión larga, track perdido) reinicia el estado del track.
    max_gap_s: float = 2.0
    # Tiempo fuera de la zona antes de dar por terminada la permanencia (histéresis).
    exit_grace_s: float = 1.0
    authorized: AuthorizedPolicy = AuthorizedPolicy.DOWNGRADE


@dataclass(frozen=True, slots=True)
class Event:
    """Evento con evidencia; solo metadata serializable, nunca frames."""

    type: EventType
    stream_id: str
    zone: str
    track_ids: tuple[int, ...]
    start_t: float
    detect_t: float
    confidence: float
    severity: Severity
    authorized: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "zone": self.zone,
            "track_ids": list(self.track_ids),
            "start_t": self.start_t,
            "detect_t": self.detect_t,
            "confidence": round(self.confidence, 4),
            "severity": self.severity.value,
            "authorized": self.authorized,
            "evidence": dict(self.evidence),
        }

    def to_envelope(self, source: str = "event") -> MetadataEnvelope:
        return MetadataEnvelope(
            source=source, stream_id=self.stream_id, payload=self.to_payload()
        )


EVENT_TOPIC = Topic.EVENTS


@dataclass(frozen=True, slots=True)
class GroundTruthEvent:
    """Evento etiquetado. `start_t` es el primer instante en que es reportable."""

    type: EventType
    stream_id: str
    zone: str
    start_t: float
    end_t: float
    track_ids: tuple[int, ...] = ()
    authorized: bool = False


def as_points(polygon: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in polygon)
