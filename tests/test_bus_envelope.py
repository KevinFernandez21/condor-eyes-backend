"""Pruebas del contrato de envelopes tipados del bus de metadata."""

import pytest

from bus import (
    SCHEMA_VERSION,
    InvalidEnvelopeError,
    InvalidTopicError,
    MetadataEnvelope,
    Topic,
    UnsupportedVersionError,
    build_error_envelope,
    envelope_from_dict,
    envelope_to_dict,
    parse_topic,
    validate_envelope,
)


def make(**kwargs):
    base = {"source": "tracker", "payload": {"track_id": 7}, "stream_id": "cam-01"}
    base.update(kwargs)
    return MetadataEnvelope(**base)


def test_envelope_defaults_are_backward_compatible():
    envelope = MetadataEnvelope(source="tracker", payload={})
    assert envelope.schema_version == SCHEMA_VERSION
    assert envelope.payload_version == 1
    assert envelope.event_id
    assert envelope.correlation_id == envelope.event_id
    assert envelope.causation_id is None


def test_event_ids_are_unique_by_default():
    assert make().event_id != make().event_id


def test_explicit_correlation_id_is_kept():
    assert make(correlation_id="corr-1").correlation_id == "corr-1"


def test_derive_keeps_correlation_and_sets_causation():
    parent = make(event_id="evt-1", correlation_id="corr-1")
    child = parent.derive(source="event", payload={"type": "x"}, suffix="events")
    assert child.correlation_id == "corr-1"
    assert child.causation_id == "evt-1"
    assert child.event_id == "evt-1/events"
    assert child.stream_id == "cam-01"


def test_parse_topic_accepts_values_and_members():
    assert parse_topic("events") is Topic.EVENTS
    assert parse_topic(Topic.TRACKS) is Topic.TRACKS


def test_parse_topic_rejects_unknown_topic_with_clear_message():
    with pytest.raises(InvalidTopicError, match="Topic desconocido.*video.frames"):
        parse_topic("video.frames")


def test_validate_rejects_unsupported_schema_version():
    with pytest.raises(UnsupportedVersionError, match="schema_version"):
        validate_envelope(Topic.EVENTS, make(schema_version=99))


def test_validate_rejects_unsupported_payload_version():
    with pytest.raises(UnsupportedVersionError, match="payload_version"):
        validate_envelope(Topic.EVENTS, make(payload_version=42))


def test_validate_rejects_empty_source_and_event_id():
    with pytest.raises(InvalidEnvelopeError, match="source"):
        validate_envelope(Topic.EVENTS, make(source=""))
    with pytest.raises(InvalidEnvelopeError, match="event_id"):
        validate_envelope(Topic.EVENTS, make(event_id=""))


@pytest.mark.parametrize(
    "payload",
    [
        {"frame": b"\x00\x01"},
        {"nested": [{"raw": bytearray(b"ab")}]},
        {"buf": memoryview(b"ab")},
    ],
)
def test_validate_rejects_binary_payloads(payload):
    with pytest.raises(InvalidEnvelopeError, match="binari|frames"):
        validate_envelope(Topic.DETECTIONS, make(payload=payload))


def test_validate_rejects_non_serializable_payload():
    with pytest.raises(InvalidEnvelopeError, match="serializable"):
        validate_envelope(Topic.DETECTIONS, make(payload={"x": object()}))


def test_validate_rejects_non_mapping_payload():
    with pytest.raises(InvalidEnvelopeError, match="Mapping"):
        validate_envelope(Topic.DETECTIONS, make(payload=[1, 2]))  # type: ignore[arg-type]


def test_dict_roundtrip_preserves_envelope():
    original = make(event_id="e1", correlation_id="c1", causation_id="e0")
    data = envelope_to_dict(Topic.TRACKS, original)
    topic, restored = envelope_from_dict(data)
    assert topic is Topic.TRACKS
    assert restored == original


def test_from_dict_rejects_invalid_topic_and_versions():
    data = envelope_to_dict(Topic.TRACKS, make())
    with pytest.raises(InvalidTopicError):
        envelope_from_dict({**data, "topic": "nope"})
    with pytest.raises(UnsupportedVersionError):
        envelope_from_dict({**data, "schema_version": 2})
    with pytest.raises(InvalidEnvelopeError, match="falta"):
        envelope_from_dict({"topic": "events"})


def test_error_envelope_links_to_failed_event():
    failed = make(event_id="evt-9", correlation_id="corr-9")
    error = build_error_envelope(
        "storage",
        RuntimeError("disco lleno"),
        failed=failed,
        failed_topic=Topic.EVENTS,
        stage="handle",
        attempts=3,
    )
    assert error.source == "storage"
    assert error.correlation_id == "corr-9"
    assert error.causation_id == "evt-9"
    assert error.payload["error_type"] == "RuntimeError"
    assert error.payload["message"] == "disco lleno"
    assert error.payload["failed_topic"] == "events"
    assert error.payload["attempts"] == 3
    validate_envelope(Topic.ERRORS, error)


def test_error_envelope_without_failed_event_gets_own_ids():
    error = build_error_envelope("ingest", ValueError("x"), stage="start")
    assert error.event_id
    assert error.causation_id is None
    assert error.payload["failed_event_id"] is None
