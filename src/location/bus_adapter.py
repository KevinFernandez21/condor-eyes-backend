"""Publicación de estimaciones por el bus tipado (`MetadataEnvelope`).

Solo viaja metadata serializable: seudónimos, zona, confianza y timestamps.
Nunca IDs de tag/persona en claro ni frames.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from bus.hub import MetadataEnvelope, MetadataHub, Topic

from .models import Evidence
from .service import LocationReport

SOURCE = "location"


def _evidence_payload(item: Evidence) -> dict[str, Any]:
    return {
        "node_id": item.node_id,
        "zone_id": item.zone_id,
        "smoothed_rssi_dbm": item.smoothed_rssi_dbm,
        "samples": item.samples,
        "first_seen_at": item.first_seen_at.isoformat(),
        "last_seen_at": item.last_seen_at.isoformat(),
    }


def to_envelope(report: LocationReport) -> MetadataEnvelope:
    """Convierte un reporte en un sobre listo para `Topic.LOCATION`."""
    est = report.estimate
    payload: dict[str, Any] = {
        "tag_ref": est.tag_ref,
        "person_ref": report.person_ref,
        "status": est.status.value,
        "zone_id": est.zone_id,
        "confidence": est.confidence,
        "unknown_reason": est.unknown_reason.value if est.unknown_reason else None,
        "degraded": est.degraded,
        "authorized": report.authorized,
        "battery_pct": est.battery_pct,
        "battery_low": est.battery_low,
        "first_evidence_at": est.first_evidence_at.isoformat()
        if est.first_evidence_at
        else None,
        "last_evidence_at": est.last_evidence_at.isoformat()
        if est.last_evidence_at
        else None,
        "computed_at": est.computed_at.isoformat(),
        "evidence": [_evidence_payload(item) for item in est.evidence],
    }
    return MetadataEnvelope(source=SOURCE, payload=payload)


async def publish_reports(hub: MetadataHub, reports: Iterable[LocationReport]) -> int:
    """Publica cada reporte en `Topic.LOCATION`; devuelve cuántos envió."""
    count = 0
    for report in reports:
        await hub.publish(Topic.LOCATION, to_envelope(report))
        count += 1
    return count
