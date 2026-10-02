"""Conversión de sobres del bus a entradas tipadas, descartando cualquier campo biométrico."""

from __future__ import annotations

from bus.hub import MetadataEnvelope

from .models import IdentityEvidence, IdentityStatus, ReidLink, ReidStatus


def identity_from_envelope(
    evidence_id: str, track_ref: str, envelope: MetadataEnvelope
) -> IdentityEvidence:
    """Toma solo estado, persona y score; ignora embeddings y el resto del payload."""
    p = envelope.payload
    return IdentityEvidence(
        evidence_id=evidence_id,
        track_ref=track_ref,
        status=IdentityStatus(p["status"]),
        observed_at=envelope.created_at,
        person_id=p.get("person_id"),
        score=float(p.get("score") or 0.0),
    )


def reid_from_envelope(evidence_id: str, envelope: MetadataEnvelope) -> ReidLink:
    """Toma solo estado, pistas y confianza de una asociación de re-ID."""
    p = envelope.payload
    return ReidLink(
        evidence_id=evidence_id,
        source_track_ref=str(p["source_track"]),
        candidate_track_ref=None
        if p.get("candidate_track") is None
        else str(p["candidate_track"]),
        status=ReidStatus(p["status"]),
        confidence=float(p.get("confidence") or 0.0),
        observed_at=envelope.created_at,
    )
