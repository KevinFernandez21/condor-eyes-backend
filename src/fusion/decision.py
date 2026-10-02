"""Registro de decisión: qué evidencia contribuyó, con qué confianza y por qué.

El registro es metadata pura: referencias a la evidencia de origen (por `evidence_id` y
marca de tiempo de la fuente), nunca embeddings, plantillas ni imágenes. El operador
humano es siempre la autoridad final, por lo que `requires_operator` no puede ser False.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from bus.hub import MetadataEnvelope


class DecisionOutcome(StrEnum):
    """Resultado de la fusión. Ninguno equivale a autorizar: decide el operador."""

    CORROBORATED = (
        "corroborated"  # rostro, tag, zona y permiso coinciden y están vigentes
    )
    ALERT = "alert"  # la evidencia contradice o no respalda la presencia
    INCONCLUSIVE = "inconclusive"  # falta, caducó o es ambigua; no se concluye nada
    NOT_RESTRICTED = "not_restricted"  # zona sin restricción: no se exige corroboración


class ReasonCode(StrEnum):
    # Evidencia consistente.
    EVIDENCE_CONSISTENT = "evidence_consistent"
    ZONE_UNRESTRICTED = "zone_unrestricted"
    # Conflictos explícitos pedidos en el issue #15.
    PERSON_WITHOUT_TAG = "person_without_tag"
    TAG_WITHOUT_PERSON = "tag_without_person"
    HIDDEN_FACE = "hidden_face"
    STALE_SENSOR = "stale_sensor"
    MULTIPLE_NEARBY_PEOPLE = "multiple_nearby_people"
    # Otros motivos de alerta o de no concluir.
    UNKNOWN_FACE = "unknown_face"
    FACE_INCONCLUSIVE = "face_inconclusive"
    IDENTITY_MISSING = "identity_missing"
    STALE_IDENTITY = "stale_identity"
    STALE_VISION = "stale_vision"
    IDENTITY_TAG_MISMATCH = "identity_tag_mismatch"
    ZONE_NOT_PERMITTED = "zone_not_permitted"
    NO_PERMISSION_RECORD = "no_permission_record"
    LOCATION_INVALID = "location_invalid"
    CLOCK_SKEW = "clock_skew"
    LOW_CONFIDENCE = "low_confidence"
    ZONE_UNKNOWN = "zone_unknown"


# Códigos que convierten el resultado en alerta; el resto de códigos de fallo da inconcluso.
ALERT_CODES = frozenset(
    {
        ReasonCode.UNKNOWN_FACE,
        ReasonCode.ZONE_NOT_PERMITTED,
        ReasonCode.IDENTITY_TAG_MISMATCH,
        ReasonCode.PERSON_WITHOUT_TAG,
    }
)


class EvidenceKind(StrEnum):
    TRACK = "track"
    IDENTITY = "identity"
    REID = "reid"
    LOCATION = "location"
    PERMISSION = "permission"


class EvidenceRole(StrEnum):
    SUPPORTS = "supports"  # contribuye a la conclusión y cuenta para la confianza
    CONFLICTS = "conflicts"  # contradice otra evidencia
    STALE = "stale"  # existe pero caducó o es inválida; no cuenta
    CONTEXT = "context"  # información relevante que no decide por sí sola


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Referencia a un mensaje de origen; `evidence_id` permite volver a su metadata."""

    evidence_id: str
    kind: EvidenceKind
    role: EvidenceRole
    observed_at: datetime | None  # marca de tiempo de la fuente, sin modificar
    confidence: float | None = None
    stream_id: str | None = None
    zone_id: str | None = None
    detail: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "role": self.role.value,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "confidence": self.confidence,
            "stream_id": self.stream_id,
            "zone_id": self.zone_id,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    decision_id: str
    evaluated_at: datetime
    outcome: DecisionOutcome
    confidence: float
    reason_codes: tuple[ReasonCode, ...]
    evidence: tuple[EvidenceRef, ...]
    track_ref: str | None = None
    stream_id: str | None = None
    zone_id: str | None = None
    person_id: str | None = None
    requires_operator: bool = True

    def __post_init__(self) -> None:
        if not self.requires_operator:
            raise ValueError("el operador humano siempre es la autoridad final")

    def to_payload(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "evaluated_at": self.evaluated_at.isoformat(),
            "outcome": self.outcome.value,
            "confidence": self.confidence,
            "reason_codes": [c.value for c in self.reason_codes],
            "track_ref": self.track_ref,
            "stream_id": self.stream_id,
            "zone_id": self.zone_id,
            "person_id": self.person_id,
            "requires_operator": True,
            "evidence": [e.to_payload() for e in self.evidence],
        }

    def to_envelope(self, source: str = "fusion") -> MetadataEnvelope:
        return MetadataEnvelope(
            source=source,
            stream_id=self.stream_id,
            payload=self.to_payload(),
            created_at=self.evaluated_at,
        )
