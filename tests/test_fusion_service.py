"""Tests de la publicación de decisiones por el bus tipado."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

from bus.hub import MetadataEnvelope, Topic
from fusion import (
    FusionConfig,
    FusionEngine,
    FusionInput,
    IdentityEvidence,
    IdentityStatus,
    InMemoryPermissions,
    LocationEvidence,
    TrackObservation,
    ZoneRule,
    publish_decisions,
)

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
FRESH = NOW - timedelta(seconds=1)


class RecordingHub:
    def __init__(self) -> None:
        self.sent: list[tuple[Topic, MetadataEnvelope]] = []

    async def publish(self, topic: Topic, message: MetadataEnvelope) -> None:
        self.sent.append((topic, message))

    def subscribe(self, topic: Topic):  # pragma: no cover - no se usa
        raise NotImplementedError


def engine() -> FusionEngine:
    return FusionEngine(
        FusionConfig(),
        zones=(ZoneRule("lab"),),
        permissions=InMemoryPermissions({"alice": frozenset({"lab"})}),
    )


def test_decisions_are_published_only_as_envelopes_on_events_topic():
    evidence = FusionInput(
        tracks=(TrackObservation("t1", "cam1/1", "cam1", "lab", FRESH, 0.9),),
        identities=(
            IdentityEvidence("i1", "cam1/1", IdentityStatus.MATCH, FRESH, "alice", 0.9),
        ),
        locations=(LocationEvidence("l1", "alice", "lab", FRESH, 0.9),),
    )
    hub = RecordingHub()
    records = engine().evaluate(evidence, NOW)
    asyncio.run(publish_decisions(hub, records))
    assert len(hub.sent) == len(records) == 1
    topic, envelope = hub.sent[0]
    assert topic is Topic.EVENTS
    assert isinstance(envelope, MetadataEnvelope)
    assert envelope.source == "fusion" and envelope.stream_id == "cam1"
    assert envelope.created_at == NOW
    payload = json.loads(json.dumps(envelope.payload))
    assert payload["requires_operator"] is True
    assert payload["decision_id"] == records[0].decision_id


def test_orphan_tag_decision_has_no_stream_but_is_published():
    evidence = FusionInput(
        locations=(LocationEvidence("l1", "alice", "lab", FRESH, 0.9),)
    )
    hub = RecordingHub()
    asyncio.run(publish_decisions(hub, engine().evaluate(evidence, NOW)))
    assert hub.sent[0][1].stream_id is None
    assert hub.sent[0][1].payload["reason_codes"] == ["tag_without_person"]
