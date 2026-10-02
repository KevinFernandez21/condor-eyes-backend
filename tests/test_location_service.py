"""Pruebas de integración: servicio, repositorio, privacidad y bus tipado."""

import asyncio
import json
import logging

import pytest
from location_support import NODE_POSITIONS, TAG, TAG2, at, make_config, obs

from bus import MetadataEnvelope, Topic
from location.bus_adapter import publish_reports, to_envelope
from location.models import EstimateStatus, RejectReason, UnknownReason
from location.payload import encode_binary, encode_json
from location.privacy import Pseudonymizer
from location.repository import (
    AccessDenied,
    InMemoryPersonnelRepository,
    PersonnelRecord,
    Principal,
)
from location.service import SERVICE_PRINCIPAL, LocationService
from location.simulator import ZoneNodeSimulator

PSEUDO = Pseudonymizer(b"clave-de-prueba")
PERSON_ID = "P-1001"
PERSON_NAME = "Ana Pérez"


def make_service(**sections):
    repo = InMemoryPersonnelRepository(PSEUDO)
    repo.enroll(
        TAG,
        PersonnelRecord(PERSON_ID, PERSON_NAME, frozenset({"lobby", "hall"})),
    )
    repo.enroll(TAG2, PersonnelRecord("P-2002", "Luis Gómez", frozenset({"lab"})))
    return LocationService(make_config(**sections), repo, PSEUDO)


def run_sim(service, sim, tag, position, times):
    results = []
    for t in times:
        for o in sim.advertise(tag, position, at(t)):
            results.append(service.ingest(o, received_at=at(t + 0.1)))
    return results


def test_end_to_end_json_ingest_and_estimate():
    service = make_service()
    sim = ZoneNodeSimulator(NODE_POSITIONS)
    for t in range(4):
        for o in sim.advertise(TAG, (1.0, 0.0), at(t)):
            assert service.ingest_json(encode_json(o), received_at=at(t + 0.1)).accepted
    (report,) = [r for r in service.reports(at(3.5)) if r.estimate.zone_id == "lobby"]
    assert report.estimate.status is EstimateStatus.LOCATED
    assert report.authorized is True
    assert report.estimate.confidence > 0.8


def test_binary_lora_ingest():
    service = make_service()
    result = service.ingest_binary(encode_binary(obs(node="N0001", t=0)), at(0.2))
    assert result.accepted


def test_malformed_payload_is_counted_not_raised():
    service = make_service()
    result = service.ingest_json("{no es json", received_at=at(0))
    assert not result.accepted and result.reason is RejectReason.MALFORMED
    assert service.stats[RejectReason.MALFORMED] == 1
    assert not service.ingest_binary(b"\x00\x01", at(0)).accepted


def test_duplicate_and_replay_are_rejected_and_counted():
    service = make_service()
    assert service.ingest(obs(seq=5, t=0), at(0)).accepted
    assert service.ingest(obs(seq=5, t=0), at(0.1)).reason is RejectReason.DUPLICATE
    assert service.ingest(obs(seq=2, t=0.2), at(0.3)).reason is RejectReason.REPLAY
    assert service.stats[RejectReason.DUPLICATE] == 1
    assert service.stats[RejectReason.REPLAY] == 1
    # El estimado no cambia con paquetes rechazados.
    (evidence,) = service.estimate(TAG, at(0.5)).evidence
    assert evidence.samples == 1


def test_stale_packet_is_rejected_and_does_not_locate():
    service = make_service()
    result = service.ingest(obs(t=0), at(60))
    assert result.reason is RejectReason.STALE
    assert service.estimate(TAG, at(60)).status is EstimateStatus.UNKNOWN


def test_clock_skew_node_is_rejected_and_other_nodes_still_locate():
    service = make_service()
    sim = ZoneNodeSimulator(NODE_POSITIONS)
    sim.set_clock_offset("N0002", 30.0)  # reloj adelantado
    sim.set_clock_offset("N0003", -60.0)  # reloj atrasado
    results = run_sim(service, sim, TAG, (1.0, 0.0), [0, 1, 2])
    reasons = {r.reason for r in results if not r.accepted}
    assert reasons == {RejectReason.CLOCK_SKEW, RejectReason.STALE}
    estimate = service.estimate(TAG, at(2.5))
    assert estimate.zone_id == "lobby"
    assert {e.node_id for e in estimate.evidence} <= {"N0001", "N0004"}


def test_small_clock_skew_within_tolerance_is_accepted():
    service = make_service()
    sim = ZoneNodeSimulator(NODE_POSITIONS)
    sim.set_clock_offset("N0001", 1.5)  # tolerancia futura = 2 s
    results = run_sim(service, sim, TAG, (1.0, 0.0), [0])
    assert any(r.accepted for r in results)


def test_missing_enrolled_tag_reported_unknown():
    service = make_service()
    run_sim(service, ZoneNodeSimulator(NODE_POSITIONS), TAG, (1.0, 0.0), [0, 1, 2])
    reports = {r.estimate.tag_ref: r for r in service.reports(at(3))}
    assert len(reports) == 2  # TAG y TAG2 enrolados
    missing = reports[PSEUDO.ref("tag", TAG2)]
    assert missing.estimate.status is EstimateStatus.UNKNOWN
    assert missing.estimate.unknown_reason is UnknownReason.TAG_MISSING
    assert missing.authorized is None


def test_unenrolled_tag_is_not_tracked():
    service = make_service()
    result = service.ingest(obs(tag="FFFFFFFFFFFF", t=0), at(0))
    assert result.reason is RejectReason.UNENROLLED_TAG
    assert "FFFFFFFFFFFF" not in service._estimator.known_tags()


def test_tag_seen_by_multiple_zones_is_resolved_by_strength():
    service = make_service()
    run_sim(service, ZoneNodeSimulator(NODE_POSITIONS), TAG, (11.0, 0.0), [0, 1, 2, 3])
    estimate = service.estimate(TAG, at(3.5))
    assert estimate.zone_id == "hall"
    assert {e.zone_id for e in estimate.evidence} >= {"hall", "lobby"}


def test_sensor_outage_yields_unknown_with_reason_then_recovers():
    service = make_service()
    sim = ZoneNodeSimulator(NODE_POSITIONS, sensitivity_dbm=-70)
    run_sim(service, sim, TAG, (1.0, 0.0), [0, 1])
    sim.set_outage("N0001", True)
    assert run_sim(service, sim, TAG, (1.0, 0.0), [20, 21]) == []
    down = service.estimate(TAG, at(45))
    assert down.status is EstimateStatus.UNKNOWN
    assert down.unknown_reason is UnknownReason.NODE_OUTAGE
    sim.set_outage("N0001", False)
    run_sim(service, sim, TAG, (1.0, 0.0), [50, 51])
    assert service.estimate(TAG, at(51.5)).zone_id == "lobby"


def test_heartbeat_marks_node_alive():
    service = make_service()
    run_sim(
        service,
        ZoneNodeSimulator(NODE_POSITIONS, sensitivity_dbm=-70),
        TAG,
        (1, 0),
        [0],
    )
    service.heartbeat("N0001", at(44))
    assert service.estimate(TAG, at(45)).unknown_reason is UnknownReason.STALE_EVIDENCE
    service.heartbeat("N9999", at(44))  # nodo desconocido: se ignora sin error


def test_authorization_follows_repository_allowed_zones():
    service = make_service()
    # TAG2 solo tiene permitido el lab, pero está en el lobby.
    run_sim(service, ZoneNodeSimulator(NODE_POSITIONS), TAG2, (1.0, 0.0), [0, 1, 2])
    (report,) = [r for r in service.reports(at(2.5)) if r.estimate.zone_id == "lobby"]
    assert report.authorized is False


def test_service_requires_repository_access():
    repo = InMemoryPersonnelRepository(PSEUDO)
    service = LocationService(
        make_config(), repo, PSEUDO, principal=Principal("sin-permiso", frozenset())
    )
    with pytest.raises(AccessDenied):
        service.ingest(obs(t=0), at(0))
    assert SERVICE_PRINCIPAL.scopes


def test_logs_never_contain_plaintext_identifiers(caplog):
    service = make_service()
    sim = ZoneNodeSimulator(NODE_POSITIONS)
    with caplog.at_level(logging.DEBUG):
        run_sim(service, sim, TAG, (1.0, 0.0), [0, 1])
        service.ingest(obs(seq=1, t=0), at(0.1))  # duplicado
        service.ingest(obs(tag="FFFFFFFFFFFF", t=0), at(0))  # no enrolado
        service.reports(at(2))
    assert caplog.records
    for secret in (TAG, TAG2, "FFFFFFFFFFFF", PERSON_ID, PERSON_NAME, "P-2002"):
        assert secret not in caplog.text


# --- bus tipado ----------------------------------------------------------


class FakeHub:
    def __init__(self):
        self.published: list[tuple[Topic, MetadataEnvelope]] = []

    async def publish(self, topic, message):
        self.published.append((topic, message))

    def subscribe(self, topic):  # pragma: no cover - no se usa
        raise NotImplementedError


def test_envelope_is_serializable_and_has_no_plaintext_ids():
    service = make_service()
    run_sim(service, ZoneNodeSimulator(NODE_POSITIONS), TAG, (1.0, 0.0), [0, 1, 2])
    (report,) = [r for r in service.reports(at(2.5)) if r.estimate.zone_id == "lobby"]
    envelope = to_envelope(report)
    assert isinstance(envelope, MetadataEnvelope)
    assert envelope.source == "location"
    text = json.dumps(envelope.payload)  # debe ser JSON puro
    for secret in (TAG, PERSON_ID, PERSON_NAME):
        assert secret not in text
    payload = envelope.payload
    assert payload["zone_id"] == "lobby"
    assert payload["status"] == "located"
    assert 0 < payload["confidence"] <= 1
    assert payload["first_evidence_at"] <= payload["last_evidence_at"]
    assert payload["person_ref"] == PSEUDO.ref("person", PERSON_ID)
    assert payload["authorized"] is True
    assert payload["evidence"][0]["node_id"] == "N0001"


def test_publish_reports_uses_location_topic():
    service = make_service()
    run_sim(service, ZoneNodeSimulator(NODE_POSITIONS), TAG, (1.0, 0.0), [0, 1])
    hub = FakeHub()
    count = asyncio.run(publish_reports(hub, service.reports(at(1.5))))
    assert count == 2
    assert {topic for topic, _ in hub.published} == {Topic.LOCATION}


def test_location_topic_is_appended_and_carries_no_video():
    assert list(Topic)[-1] is Topic.LOCATION
    assert "frame" not in Topic.LOCATION.value and "video" not in Topic.LOCATION.value


@pytest.mark.parametrize(
    "raw",
    ["[" * 100_000, '{"a":' * 100_000, None, 42, b"\xff\xfe", "x" * 5000],
    ids=["corchetes", "objetos", "none", "int", "utf8-invalido", "grande"],
)
def test_hostile_json_never_raises_and_is_counted_as_malformed(raw):
    service = make_service()
    result = service.ingest_json(raw, at(0))
    assert not result.accepted and result.reason is RejectReason.MALFORMED
    assert service.stats[RejectReason.MALFORMED] == 1


@pytest.mark.parametrize(
    "raw", ["x" * 26, None, 7, b""], ids=["str", "none", "int", "vacio"]
)
def test_hostile_binary_never_raises(raw):
    service = make_service()
    result = service.ingest_binary(raw, at(0))
    assert result.reason is RejectReason.MALFORMED


def test_directly_built_lowercase_tag_is_normalized():
    service = make_service()
    lower = obs(tag=TAG.lower(), t=0)
    assert service.ingest(lower, at(0.1)).accepted
    assert service.estimate(TAG, at(0.5)).zone_id == "lobby"
    assert TAG.lower() not in service._estimator.known_tags()


def test_every_published_envelope_payload_is_strict_json():
    service = make_service()
    run_sim(service, ZoneNodeSimulator(NODE_POSITIONS), TAG, (1.0, 0.0), [0, 1, 2])
    reports = service.reports(at(2.5)) + service.reports(at(100))
    assert {r.estimate.status for r in reports} == {
        EstimateStatus.LOCATED,
        EstimateStatus.UNKNOWN,
    }

    def check(value):
        assert value is None or type(value) in (str, int, float, bool, list, dict)
        if isinstance(value, dict):
            assert all(type(k) is str for k in value)
            [check(v) for v in value.values()]
        elif isinstance(value, list):
            [check(v) for v in value]

    for report in reports:
        payload = to_envelope(report).payload
        json.dumps(payload, allow_nan=False)
        check(dict(payload))
