"""Publicación de decisiones por el bus tipado (solo sobres de metadata)."""

from __future__ import annotations

from collections.abc import Iterable

from bus.hub import MetadataHub, Topic

from .decision import DecisionRecord


async def publish_decisions(
    hub: MetadataHub, records: Iterable[DecisionRecord], source: str = "fusion"
) -> None:
    """Publica cada decisión en `Topic.EVENTS` como `MetadataEnvelope`, en orden."""
    for record in records:
        await hub.publish(Topic.EVENTS, record.to_envelope(source))
