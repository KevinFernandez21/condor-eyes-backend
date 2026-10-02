"""Conversión de sobres del bus a entradas tipadas, descartando cualquier campo biométrico.

Un payload defectuoso nunca lanza: se convierte en evidencia no concluyente
(`INVALID_INPUT` / `INCONCLUSIVE`), que la fusión jamás cuenta a favor.
"""

from __future__ import annotations

import math
from typing import Any

from bus.hub import MetadataEnvelope

from .models import IdentityEvidence, IdentityStatus, ReidLink, ReidStatus


def _unit_float(value: Any) -> float:
    """Número finito en [0, 1]; cualquier otra cosa lanza ValueError/TypeError."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("se esperaba un número")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("fuera de [0, 1]")
    return number


def identity_from_envelope(
    evidence_id: str, track_ref: str, envelope: MetadataEnvelope
) -> IdentityEvidence:
    """Toma solo estado, persona y score; ignora embeddings y el resto del payload."""
    p = envelope.payload
    try:
        status = IdentityStatus(p["status"])
        person = p.get("person_id")
        if person is not None and not isinstance(person, str):
            raise TypeError("person_id inválido")
        score = _unit_float(p.get("score") or 0.0)
    except (KeyError, TypeError, ValueError):
        return IdentityEvidence(
            evidence_id, track_ref, IdentityStatus.INVALID_INPUT, envelope.created_at
        )
    return IdentityEvidence(
        evidence_id, track_ref, status, envelope.created_at, person, score
    )


def reid_from_envelope(evidence_id: str, envelope: MetadataEnvelope) -> ReidLink:
    """Toma solo estado, pistas y confianza de una asociación de re-ID."""
    p = envelope.payload
    try:
        status = ReidStatus(p["status"])
        source = str(p["source_track"])
        candidate = p.get("candidate_track")
        candidate_ref = None if candidate is None else str(candidate)
        confidence = _unit_float(p.get("confidence") or 0.0)
    except (KeyError, TypeError, ValueError):
        return ReidLink(
            evidence_id,
            str(p.get("source_track", "")),
            None,
            ReidStatus.INCONCLUSIVE,
            0.0,
            envelope.created_at,
        )
    return ReidLink(
        evidence_id, source, candidate_ref, status, confidence, envelope.created_at
    )
