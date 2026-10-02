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


def _assert_json_primitives(value, path="payload"):
    """Mismo contrato estricto que el bus: solo primitivas JSON y claves str."""
    if value is None or type(value) in (str, int, float, bool):
        return
    if type(value) in (list, tuple):
        for i, item in enumerate(value):
            _assert_json_primitives(item, f"{path}[{i}]")
        return
    if type(value) is dict:
        for k, v in value.items():
            assert type(k) is str, f"{path}: clave no str {k!r}"
            _assert_json_primitives(v, f"{path}.{k}")
        return
    raise AssertionError(f"{path}: tipo no permitido {type(value).__name__}")


def test_every_published_envelope_payload_is_strict_json():
    ids = (
        IdentityEvidence("i1", "cam1/1", IdentityStatus.MATCH, FRESH, "alice", 0.9),
        IdentityEvidence("i2", "cam1/2", IdentityStatus.NO_FACE, FRESH),
    )
    evidence = FusionInput(
        tracks=(
            TrackObservation("t1", "cam1/1", "cam1", "lab", FRESH, 0.9),
            TrackObservation("t2", "cam1/2", "cam1", "lab", FRESH, 0.9),
            TrackObservation("t3", "cam1/3", "cam1", "nowhere", FRESH, 0.9),
        ),
        identities=ids,
        locations=(
            LocationEvidence("l1", "alice", "lab", FRESH, 0.9),
            LocationEvidence("l2", "bob", "lab", FRESH, 0.9, valid=False),
            LocationEvidence("l3", "carol", "lab", NOW - timedelta(seconds=90), 0.9),
        ),
    )
    hub = RecordingHub()
    records = engine().evaluate(evidence, NOW)
    assert len(records) >= 3
    asyncio.run(publish_decisions(hub, records))
    assert len(hub.sent) == len(records)
    for _, envelope in hub.sent:
        _assert_json_primitives(envelope.payload)
        json.dumps(envelope.payload)  # no debe lanzar
        assert isinstance(envelope.payload["evaluated_at"], str)
        for ref in envelope.payload["evidence"]:
            assert ref["observed_at"] is None or isinstance(ref["observed_at"], str)
