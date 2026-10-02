"""Tests del motor de fusión: casos de fallo, determinismo y trazabilidad (issue #15)."""

from __future__ import annotations

import itertools
import json
import random
from datetime import UTC, datetime, timedelta

import pytest

from bus.hub import MetadataEnvelope
from fusion import (
    DecisionOutcome,
    DecisionRecord,
    FusionConfig,
    FusionEngine,
    FusionInput,
    IdentityEvidence,
    IdentityStatus,
    InMemoryPermissions,
    LocationEvidence,
    ReasonCode,
    ReidLink,
    ReidStatus,
    TrackObservation,
    ZoneRule,
    identity_from_envelope,
    reid_from_envelope,
)

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
FRESH = NOW - timedelta(seconds=1)
LAB, HALL = "lab", "hall"


def ago(seconds: float) -> datetime:
    return NOW - timedelta(seconds=seconds)


def make_engine(**cfg) -> FusionEngine:
    return FusionEngine(
        FusionConfig(**cfg),
        zones=(ZoneRule(LAB, restricted=True), ZoneRule(HALL, restricted=False)),
        permissions=InMemoryPermissions(
            {"alice": frozenset({LAB, HALL}), "bob": frozenset({HALL})}
        ),
    )


def track(ref="cam1/1", zone=LAB, at=FRESH, conf=0.9, eid=None) -> TrackObservation:
    return TrackObservation(eid or f"trk-{ref}", ref, ref.split("/")[0], zone, at, conf)


def ident(
    ref="cam1/1", status=IdentityStatus.MATCH, person="alice", score=0.9, at=FRESH
):
    return IdentityEvidence(f"id-{ref}", ref, status, at, person, score)


def loc(person="alice", zone=LAB, at=FRESH, conf=0.9, valid=True, eid=None):
    return LocationEvidence(eid or f"loc-{person}", person, zone, at, conf, valid)


def one(engine: FusionEngine, evidence: FusionInput) -> DecisionRecord:
    records = engine.evaluate(evidence, NOW)
    assert len(records) == 1
    return records[0]


# --- caso feliz y autoridad del operador -------------------------------------


def test_consistent_evidence_is_corroborated_with_min_confidence():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(conf=0.95),),
            identities=(ident(score=0.8),),
            locations=(loc(conf=0.7),),
        ),
    )
    assert rec.outcome is DecisionOutcome.CORROBORATED
    assert rec.reason_codes == (ReasonCode.EVIDENCE_CONSISTENT,)
    assert rec.confidence == pytest.approx(0.7)
    assert rec.person_id == "alice" and rec.zone_id == LAB
    assert {r.evidence_id for r in rec.evidence} >= {
        "trk-cam1/1",
        "id-cam1/1",
        "loc-alice",
    }


def test_operator_is_always_final_authority():
    rec = one(
        make_engine(),
        FusionInput(tracks=(track(),), identities=(ident(),), locations=(loc(),)),
    )
    assert rec.requires_operator is True
    assert rec.to_payload()["requires_operator"] is True
    with pytest.raises(ValueError, match="operador"):
        DecisionRecord(
            decision_id="d",
            evaluated_at=NOW,
            outcome=DecisionOutcome.CORROBORATED,
            confidence=1.0,
            reason_codes=(ReasonCode.EVIDENCE_CONSISTENT,),
            evidence=(),
            requires_operator=False,
        )


# --- casos de fallo de la presentación ---------------------------------------


def test_person_without_tag_and_unknown_face_is_alert():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),),
            identities=(ident(status=IdentityStatus.UNKNOWN, person=None, score=0.2),),
        ),
    )
    assert rec.outcome is DecisionOutcome.ALERT
    assert set(rec.reason_codes) == {
        ReasonCode.UNKNOWN_FACE,
        ReasonCode.PERSON_WITHOUT_TAG,
    }


def test_matched_person_without_tag_is_not_corroborated():
    rec = one(make_engine(), FusionInput(tracks=(track(),), identities=(ident(),)))
    assert rec.outcome is DecisionOutcome.ALERT
    assert ReasonCode.PERSON_WITHOUT_TAG in rec.reason_codes


def test_hidden_face_with_tag_present_is_inconclusive():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),),
            identities=(ident(status=IdentityStatus.NO_FACE, person=None, score=0.0),),
            locations=(loc(),),
        ),
    )
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.HIDDEN_FACE in rec.reason_codes
    assert rec.person_id is None


def test_stale_sensor_never_becomes_authorized():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),), identities=(ident(),), locations=(loc(at=ago(60)),)
        ),
    )
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.STALE_SENSOR in rec.reason_codes
    assert ReasonCode.PERSON_WITHOUT_TAG not in rec.reason_codes
    stale = [r for r in rec.evidence if r.evidence_id == "loc-alice"]
    assert stale and stale[0].observed_at == ago(60)


def test_stale_sensor_in_zone_explains_missing_tag_for_unidentified_person():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),),
            identities=(ident(status=IdentityStatus.NO_FACE, person=None, score=0.0),),
            locations=(loc(at=ago(60)),),
        ),
    )
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.STALE_SENSOR in rec.reason_codes
    assert ReasonCode.PERSON_WITHOUT_TAG not in rec.reason_codes


def test_tag_without_person_is_inconclusive_and_has_no_track():
    rec = one(make_engine(), FusionInput(locations=(loc(),)))
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert rec.reason_codes == (ReasonCode.TAG_WITHOUT_PERSON,)
    assert rec.track_ref is None and rec.person_id == "alice"


def test_tag_with_unidentified_track_in_zone_is_not_orphan():
    records = make_engine().evaluate(
        FusionInput(
            tracks=(track(),),
            identities=(ident(status=IdentityStatus.NO_FACE, person=None, score=0.0),),
            locations=(loc(),),
        ),
        NOW,
    )
    assert all(ReasonCode.TAG_WITHOUT_PERSON not in r.reason_codes for r in records)


def test_multiple_nearby_people_cannot_be_assigned_to_one_tag():
    no_face = {"status": IdentityStatus.NO_FACE, "person": None, "score": 0.0}
    records = make_engine().evaluate(
        FusionInput(
            tracks=(track("cam1/1"), track("cam1/2")),
            identities=(ident("cam1/1", **no_face), ident("cam1/2", **no_face)),
            locations=(loc(),),
        ),
        NOW,
    )
    assert len(records) == 2
    for rec in records:
        assert rec.outcome is DecisionOutcome.INCONCLUSIVE
        assert ReasonCode.MULTIPLE_NEARBY_PEOPLE in rec.reason_codes


def test_multiple_people_do_not_block_face_corroborated_track():
    records = make_engine().evaluate(
        FusionInput(
            tracks=(track("cam1/1"), track("cam1/2")),
            identities=(
                ident("cam1/1"),
                ident("cam1/2", status=IdentityStatus.UNKNOWN, person=None, score=0.1),
            ),
            locations=(loc(),),
        ),
        NOW,
    )
    by_ref = {r.track_ref: r for r in records}
    assert by_ref["cam1/1"].outcome is DecisionOutcome.CORROBORATED
    assert by_ref["cam1/2"].outcome is DecisionOutcome.ALERT


# --- conflictos de identidad y permisos --------------------------------------


def test_identity_vs_tag_zone_mismatch_is_alert():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),), identities=(ident(),), locations=(loc(zone=HALL),)
        ),
    )
    assert rec.outcome is DecisionOutcome.ALERT
    assert ReasonCode.IDENTITY_TAG_MISMATCH in rec.reason_codes


def test_zone_not_permitted_is_alert_even_with_consistent_evidence():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),),
            identities=(ident(person="bob"),),
            locations=(loc(person="bob"),),
        ),
    )
    assert rec.outcome is DecisionOutcome.ALERT
    assert ReasonCode.ZONE_NOT_PERMITTED in rec.reason_codes


def test_missing_permission_record_is_never_authorized():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),),
            identities=(ident(person="ghost"),),
            locations=(loc(person="ghost"),),
        ),
    )
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.NO_PERMISSION_RECORD in rec.reason_codes


def test_invalid_location_packet_is_not_usable():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),), identities=(ident(),), locations=(loc(valid=False),)
        ),
    )
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.LOCATION_INVALID in rec.reason_codes


def test_future_timestamp_is_clock_skew_not_fresh():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),),
            identities=(ident(),),
            locations=(loc(at=NOW + timedelta(seconds=30)),),
        ),
    )
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.CLOCK_SKEW in rec.reason_codes


def test_missing_identity_is_inconclusive():
    rec = one(make_engine(), FusionInput(tracks=(track(),), locations=(loc(),)))
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.IDENTITY_MISSING in rec.reason_codes


def test_stale_identity_is_inconclusive():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),), identities=(ident(at=ago(120)),), locations=(loc(),)
        ),
    )
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.STALE_IDENTITY in rec.reason_codes


def test_low_identity_score_match_is_inconclusive():
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),), identities=(ident(score=0.4),), locations=(loc(),)
        ),
    )
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.LOW_CONFIDENCE in rec.reason_codes


def test_low_combined_confidence_downgrades_to_inconclusive():
    rec = one(
        make_engine(min_decision_confidence=0.95),
        FusionInput(tracks=(track(),), identities=(ident(),), locations=(loc(),)),
    )
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.LOW_CONFIDENCE in rec.reason_codes


def test_stale_track_is_inconclusive():
    records = make_engine().evaluate(
        FusionInput(
            tracks=(track(at=ago(60)),),
            identities=(ident(at=ago(60)),),
            locations=(loc(),),
        ),
        NOW,
    )
    rec = next(r for r in records if r.track_ref == "cam1/1")
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.STALE_VISION in rec.reason_codes


def test_unrestricted_zone_requires_no_corroboration():
    rec = one(make_engine(), FusionInput(tracks=(track(zone=HALL),)))
    assert rec.outcome is DecisionOutcome.NOT_RESTRICTED
    assert rec.reason_codes == (ReasonCode.ZONE_UNRESTRICTED,)


def test_unknown_zone_is_inconclusive():
    rec = one(make_engine(), FusionInput(tracks=(track(zone="nowhere"),)))
    assert rec.outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.ZONE_UNKNOWN in rec.reason_codes


# --- re-ID -------------------------------------------------------------------


def reid(status=ReidStatus.LINKED, conf=0.9, at=FRESH, cand="cam1/1", src="cam2/5"):
    return ReidLink("reid-1", src, cand, status, conf, at)


def reid_input(link: ReidLink) -> FusionInput:
    return FusionInput(
        tracks=(track("cam1/1"), track("cam2/5")),
        identities=(ident("cam1/1"),),
        reids=(link,),
        locations=(loc(),),
    )


def test_linked_reid_propagates_identity_to_other_camera_track():
    records = make_engine().evaluate(reid_input(reid(conf=0.8)), NOW)
    rec = {r.track_ref: r for r in records}["cam2/5"]
    assert rec.outcome is DecisionOutcome.CORROBORATED
    assert rec.confidence == pytest.approx(0.8)
    assert "reid-1" in {r.evidence_id for r in rec.evidence}


@pytest.mark.parametrize(
    "link",
    [
        reid(status=ReidStatus.INCONCLUSIVE),
        reid(status=ReidStatus.NO_CANDIDATE, cand=None),
        reid(conf=0.2),
        reid(at=ago(300)),
    ],
)
def test_unusable_reid_does_not_propagate_identity(link):
    rec = {r.track_ref: r for r in make_engine().evaluate(reid_input(link), NOW)}[
        "cam2/5"
    ]
    assert rec.outcome is not DecisionOutcome.CORROBORATED


# --- determinismo, trazabilidad, privacidad ----------------------------------


def busy_input() -> FusionInput:
    return FusionInput(
        tracks=(track("cam1/1"), track("cam1/2"), track("cam2/1", zone=HALL)),
        identities=(
            ident("cam1/1"),
            ident("cam1/2", status=IdentityStatus.NO_FACE, person=None),
        ),
        reids=(reid(),),
        locations=(loc(), loc("bob", zone=LAB, at=ago(90))),
    )


def test_same_ordered_evidence_gives_identical_records():
    engine = make_engine()
    first = engine.evaluate(busy_input(), NOW)
    assert first == engine.evaluate(busy_input(), NOW)
    assert first == make_engine().evaluate(busy_input(), NOW)


def test_result_does_not_depend_on_arrival_order():
    base = busy_input()
    expected = make_engine().evaluate(base, NOW)
    rng = random.Random(7)
    for _ in range(10):
        shuffled = FusionInput(
            *(
                tuple(rng.sample(list(part), len(part)))
                for part in (base.tracks, base.identities, base.reids, base.locations)
            )
        )
        assert make_engine().evaluate(shuffled, NOW) == expected


def test_decision_id_changes_when_evidence_changes():
    engine = make_engine()
    a = one(
        engine,
        FusionInput(tracks=(track(),), identities=(ident(),), locations=(loc(),)),
    )
    b = one(
        engine,
        FusionInput(
            tracks=(track(),), identities=(ident(),), locations=(loc(at=ago(2)),)
        ),
    )
    assert a.decision_id != b.decision_id


def test_every_evidence_ref_traces_to_input_ids_and_timestamps():
    evidence = busy_input()
    sources = {
        e.evidence_id: e.observed_at
        for e in (
            *evidence.tracks,
            *evidence.identities,
            *evidence.reids,
            *evidence.locations,
        )
    }
    for rec in make_engine().evaluate(evidence, NOW):
        for ref in rec.evidence:
            if ref.kind.value == "permission":
                continue
            assert ref.evidence_id in sources
            assert ref.observed_at == sources[ref.evidence_id]


def test_payload_is_json_serializable_and_has_no_biometric_data():
    rec = one(
        make_engine(),
        FusionInput(tracks=(track(),), identities=(ident(),), locations=(loc(),)),
    )
    text = json.dumps(rec.to_payload())
    for forbidden in ("embedding", "template", "image", "frame", "bbox"):
        assert forbidden not in text
    payload = json.loads(text)
    assert payload["outcome"] == "corroborated"
    assert payload["evidence"][0]["evidence_id"]


def test_identity_adapter_drops_biometric_fields_from_envelope():
    envelope = MetadataEnvelope(
        source="identity",
        stream_id="cam1",
        payload={
            "status": "match",
            "person_id": "alice",
            "score": 0.9,
            "requires_operator": True,
            "embedding": [0.1, 0.2],
            "evidence": {"quality": 0.8},
        },
        created_at=FRESH,
    )
    ev = identity_from_envelope("env-1", "cam1/1", envelope)
    assert ev.status is IdentityStatus.MATCH and ev.person_id == "alice"
    assert ev.observed_at == FRESH and ev.evidence_id == "env-1"
    assert "embedding" not in repr(ev)


# --- barrido: nada obsoleto o ausente se autoriza ----------------------------


def test_only_fully_fresh_evidence_can_corroborate():
    """Combina frescura/ausencia de cada fuente: solo todas frescas corroboran."""
    engine = make_engine()
    states = ("fresh", "stale", "missing")
    for t_state, i_state, l_state in itertools.product(states, repeat=3):
        age = {"fresh": FRESH, "stale": ago(600)}
        tracks = (track(at=age[t_state]),) if t_state != "missing" else ()
        ids = (ident(at=age[i_state]),) if i_state != "missing" else ()
        locs = (loc(at=age[l_state]),) if l_state != "missing" else ()
        records = engine.evaluate(FusionInput(tracks, ids, (), locs), NOW)
        corroborated = any(r.outcome is DecisionOutcome.CORROBORATED for r in records)
        assert corroborated == (t_state == i_state == l_state == "fresh"), (
            t_state,
            i_state,
            l_state,
        )


def test_empty_input_produces_no_decisions():
    assert make_engine().evaluate(FusionInput(), NOW) == ()


def test_nan_config_cannot_make_old_evidence_corroborated():
    nan = float("nan")
    with pytest.raises(ValueError):
        make_engine(
            track_max_age_s=nan,
            identity_max_age_s=nan,
            location_max_age_s=nan,
            max_clock_skew_s=nan,
            time_window_s=nan,
        )


# --- revisión: lecturas posteriores, duplicados y entradas defectuosas ------


@pytest.mark.parametrize(
    ("newer", "code"),
    [
        (
            loc(zone=HALL, at=ago(1), valid=False, eid="loc-new"),
            ReasonCode.LOCATION_INVALID,
        ),
        (
            loc(zone=HALL, at=ago(1), conf=0.1, eid="loc-new"),
            ReasonCode.LOW_CONFIDENCE,
        ),
        (
            loc(zone=LAB, at=ago(1), valid=False, eid="loc-new"),
            ReasonCode.LOCATION_INVALID,
        ),
        (
            loc(zone=LAB, at=ago(1), conf=0.1, eid="loc-new"),
            ReasonCode.LOW_CONFIDENCE,
        ),
        (
            loc(zone=HALL, at=ago(1), eid="loc-new"),
            ReasonCode.IDENTITY_TAG_MISMATCH,
        ),
    ],
)
def test_newer_bad_or_conflicting_reading_degrades_corroboration(newer, code):
    rec = one(
        make_engine(),
        FusionInput(
            tracks=(track(),),
            identities=(ident(),),
            locations=(loc(at=ago(5), eid="loc-old"), newer),
        ),
    )
    assert rec.outcome is not DecisionOutcome.CORROBORATED
    assert code in rec.reason_codes
    assert "loc-new" in {r.evidence_id for r in rec.evidence}


def test_same_track_ref_with_same_zone_keeps_latest_single_record():
    records = make_engine().evaluate(
        FusionInput(
            tracks=(track(eid="old", at=ago(2)), track(eid="new", at=ago(1))),
            identities=(ident(),),
            locations=(loc(),),
        ),
        NOW,
    )
    assert len(records) == 1
    assert "new" in {r.evidence_id for r in records[0].evidence}


def test_same_track_ref_in_different_zones_is_inconclusive():
    records = make_engine().evaluate(
        FusionInput(
            tracks=(track(eid="a", zone=LAB), track(eid="b", zone=HALL)),
            identities=(ident(),),
            locations=(loc(),),
        ),
        NOW,
    )
    recs = [r for r in records if r.track_ref == "cam1/1"]
    assert len(recs) == 1
    assert recs[0].outcome is DecisionOutcome.INCONCLUSIVE
    assert ReasonCode.DUPLICATE_TRACK in recs[0].reason_codes


def test_same_person_on_two_tracks_is_not_corroborated():
    records = make_engine().evaluate(
        FusionInput(
            tracks=(track("cam1/1"), track("cam1/2")),
            identities=(ident("cam1/1"), ident("cam1/2")),
            locations=(loc(),),
        ),
        NOW,
    )
    assert len(records) == 2
    for rec in records:
        assert rec.outcome is DecisionOutcome.INCONCLUSIVE
        assert ReasonCode.DUPLICATE_IDENTITY in rec.reason_codes


def test_naive_now_gives_clear_error():
    with pytest.raises(ValueError, match="zona horaria"):
        make_engine().evaluate(FusionInput(), NOW.replace(tzinfo=None))


class _StubEnvelope:
    """Sobre mínimo (duck-typing): permite probar payloads que un MetadataEnvelope
    estricto (NaN, tipos no JSON) rechazaría al construirse."""

    def __init__(self, source, payload):
        self.source = source
        self.payload = payload
        self.stream_id = None
        self.created_at = FRESH


def _stub_envelope(source, payload):
    return _StubEnvelope(source, payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status": "match", "person_id": "alice", "score": "abc"},
        {"status": "match", "person_id": "alice", "score": float("nan")},
        {"status": "match", "person_id": "alice", "score": 7},
        {"status": "nonsense"},
    ],
)
def test_identity_adapter_returns_invalid_input_instead_of_raising(payload):
    env = _stub_envelope("identity", payload)
    ev = identity_from_envelope("env-1", "cam1/1", env)
    assert ev.status is IdentityStatus.INVALID_INPUT and ev.person_id is None
    assert ev.score == 0.0 and ev.evidence_id == "env-1"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "status": "linked",
            "source_track": "a",
            "candidate_track": "b",
            "confidence": float("nan"),
        },
        {"status": "linked", "candidate_track": "b", "confidence": 0.9},
        {"status": "weird", "source_track": "a"},
    ],
)
def test_reid_adapter_returns_inconclusive_instead_of_raising(payload):
    env = _stub_envelope("reid", payload)
    link = reid_from_envelope("env-2", env)
    assert link.status is ReidStatus.INCONCLUSIVE and link.confidence == 0.0
