"""Modelos de dominio de la localización de personal por zonas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Channel(StrEnum):
    """Canal por el que llegó la observación al componente de localización."""

    BLE = "ble"
    LORA = "lora"


@dataclass(frozen=True, slots=True)
class TagObservation:
    """Una lectura de un tag de personal hecha por un nodo de zona.

    `tag_id` es un identificador de persona indirecto: se excluye del `repr`
    para que no aparezca por accidente en logs o trazas de pytest.
    """

    tag_id: str = field(repr=False)
    node_id: str
    rssi_dbm: int
    timestamp: datetime
    sequence: int
    battery_pct: int | None = None
    channel: Channel = Channel.BLE


class RejectReason(StrEnum):
    """Motivo por el que una observación se descarta."""

    MALFORMED = "malformed"
    UNENROLLED_TAG = "unenrolled_tag"
    UNKNOWN_NODE = "unknown_node"
    IMPOSSIBLE_VALUE = "impossible_value"
    STALE = "stale"
    CLOCK_SKEW = "clock_skew"
    DUPLICATE = "duplicate"
    REPLAY = "replay"
    IMPOSSIBLE_TRANSITION = "impossible_transition"


class EstimateStatus(StrEnum):
    """Estado de la estimación de zona."""

    LOCATED = "located"
    UNKNOWN = "unknown"


class UnknownReason(StrEnum):
    """Por qué no hay zona para un tag."""

    TAG_MISSING = "tag_missing"
    STALE_EVIDENCE = "stale_evidence"
    NODE_OUTAGE = "node_outage"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Evidencia de un nodo que respalda una estimación."""

    node_id: str
    zone_id: str
    smoothed_rssi_dbm: float
    samples: int
    first_seen_at: datetime
    last_seen_at: datetime


@dataclass(frozen=True, slots=True)
class ZoneEstimate:
    """Estimación de zona de un tag. Solo contiene el seudónimo del tag."""

    tag_ref: str
    status: EstimateStatus
    zone_id: str | None
    confidence: float
    computed_at: datetime
    first_evidence_at: datetime | None = None
    last_evidence_at: datetime | None = None
    evidence: tuple[Evidence, ...] = ()
    unknown_reason: UnknownReason | None = None
    degraded: bool = False
    battery_pct: int | None = None
    battery_low: bool = False
