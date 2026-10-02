"""Pruebas del estimador de zona y del simulador de nodos."""

import pytest
from location_support import NODE_POSITIONS, TAG, TAG2, at, make_config, obs

from location.estimator import ZoneEstimator
from location.models import EstimateStatus, UnknownReason
from location.privacy import Pseudonymizer
from location.simulator import ZoneNodeSimulator

PSEUDO = Pseudonymizer(b"clave-de-prueba")


def make_estimator(**sections):
    return ZoneEstimator(make_config(**sections), PSEUDO)


def feed(estimator, sim, tag, position, times, battery=90):
    for t in times:
        for o in sim.advertise(tag, position, at(t), battery_pct=battery):
            estimator.add(o)


# --- simulador ---------------------------------------------------------------


def test_simulator_is_deterministic_for_a_seed():
    a = ZoneNodeSimulator(NODE_POSITIONS, seed=7, noise_db=3.0)
    b = ZoneNodeSimulator(NODE_POSITIONS, seed=7, noise_db=3.0)
    c = ZoneNodeSimulator(NODE_POSITIONS, seed=8, noise_db=3.0)
    run = lambda s: [s.advertise(TAG, (1.0, 0.0), at(i)) for i in range(5)]
    assert run(a) == run(b)
    assert run(a) != run(c)


def test_simulator_closer_node_has_stronger_rssi_and_shared_sequence():
    sim = ZoneNodeSimulator(NODE_POSITIONS)
    heard = {o.node_id: o for o in sim.advertise(TAG, (1.0, 0.0), at(0))}
    assert heard["N0001"].rssi_dbm > heard["N0002"].rssi_dbm > heard["N0004"].rssi_dbm
    assert len({o.sequence for o in heard.values()}) == 1
    assert (
        sim.advertise(TAG, (1.0, 0.0), at(1))[0].sequence == heard["N0001"].sequence + 1
    )


def test_simulator_sensitivity_outage_and_clock_offset():
    sim = ZoneNodeSimulator(NODE_POSITIONS, sensitivity_dbm=-70)
    assert {o.node_id for o in sim.advertise(TAG, (1.0, 0.0), at(0))} == {"N0001"}
    sim.set_outage("N0001", True)
    assert sim.advertise(TAG, (1.0, 0.0), at(1)) == []
    sim.set_outage("N0001", False)
    sim.set_clock_offset("N0001", 30.0)
    (skewed,) = sim.advertise(TAG, (1.0, 0.0), at(2))
    assert (skewed.timestamp - at(2)).total_seconds() == 30.0


# --- estimación --------------------------------------------------------------


def test_stable_tag_is_located_with_high_confidence_and_evidence_times():
    est = make_estimator()
    sim = ZoneNodeSimulator(NODE_POSITIONS)
    feed(est, sim, TAG, (1.0, 0.0), [0, 1, 2, 3, 4])
    result = est.estimate(TAG, at(4.5))
    assert result.status is EstimateStatus.LOCATED
    assert result.zone_id == "lobby"
    assert result.confidence > 0.8
    assert result.first_evidence_at == at(0)
    assert result.last_evidence_at == at(4)
    assert result.computed_at == at(4.5)
    assert {e.node_id for e in result.evidence} >= {"N0001"}
    assert TAG not in repr(result)
    assert result.tag_ref == PSEUDO.ref("tag", TAG)


def test_missing_tag_is_unknown_not_an_error():
    est = make_estimator()
    result = est.estimate(TAG, at(0))
    assert result.status is EstimateStatus.UNKNOWN
    assert result.zone_id is None
    assert result.confidence == 0.0
    assert result.unknown_reason is UnknownReason.TAG_MISSING
    assert result.evidence == ()


def test_evidence_expires_after_freshness_window():
    est = make_estimator()
    sim = ZoneNodeSimulator(NODE_POSITIONS, sensitivity_dbm=-70)
    feed(est, sim, TAG, (1.0, 0.0), [0, 1, 2])
    assert est.estimate(TAG, at(16)).status is EstimateStatus.LOCATED
    late = est.estimate(TAG, at(18))  # evidence_max_age_s = 15, última muestra t=2
    assert late.status is EstimateStatus.UNKNOWN
    assert late.unknown_reason is UnknownReason.STALE_EVIDENCE


def test_tag_seen_by_two_zones_picks_strongest_with_lower_confidence():
    sim = ZoneNodeSimulator(NODE_POSITIONS)
    clear, ambiguous = make_estimator(), make_estimator()
    feed(clear, sim, TAG, (1.0, 0.0), [0, 1, 2, 3])
    sim2 = ZoneNodeSimulator(NODE_POSITIONS)
    feed(ambiguous, sim2, TAG, (4.5, 0.0), [0, 1, 2, 3])  # casi equidistante
    clear_r, amb_r = clear.estimate(TAG, at(3)), ambiguous.estimate(TAG, at(3))
    assert amb_r.status is EstimateStatus.LOCATED
    assert len({e.zone_id for e in amb_r.evidence}) >= 2
    assert amb_r.confidence < clear_r.confidence
    assert amb_r.confidence < 0.6


def test_hysteresis_prevents_zone_flapping():
    est = make_estimator()
    sim = ZoneNodeSimulator(NODE_POSITIONS)
    feed(est, sim, TAG, (1.0, 0.0), [0, 1, 2, 3])
    assert est.estimate(TAG, at(3)).zone_id == "lobby"
    # Posición ligeramente más cerca del pasillo: ventaja < hysteresis_db.
    feed(est, sim, TAG, (5.4, 0.0), [4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert est.estimate(TAG, at(12)).zone_id == "lobby"
    # Ya claramente en el pasillo.
    feed(est, sim, TAG, (11.0, 0.0), list(range(13, 25)))
    assert est.estimate(TAG, at(24)).zone_id == "hall"


def test_smoothing_dampens_an_outlier():
    smooth = make_estimator(smoothing={"alpha": 0.3})
    raw = make_estimator(smoothing={"alpha": 1.0})
    for est in (smooth, raw):
        for i, rssi in enumerate([-60, -60, -60, -60, -90]):
            est.add(obs(rssi=rssi, t=i, seq=i + 1))
    (s,) = smooth.estimate(TAG, at(4)).evidence
    (r,) = raw.estimate(TAG, at(4)).evidence
    assert r.smoothed_rssi_dbm == -90
    assert -75 < s.smoothed_rssi_dbm < -60


def test_samples_outside_smoothing_window_are_ignored():
    est = make_estimator()  # window_s = 8, evidence_max_age_s = 15
    est.add(obs(rssi=-90, t=0, seq=1))
    est.add(obs(rssi=-60, t=12, seq=2))
    (e,) = est.estimate(TAG, at(12)).evidence
    assert e.smoothed_rssi_dbm == -60
    assert e.samples == 1


def test_degraded_when_neighbour_zone_nodes_are_offline():
    est = make_estimator()
    sim = ZoneNodeSimulator(NODE_POSITIONS)
    sim.set_outage("N0001", True)
    est.touch_node("N0001", at(-100))  # visto hace mucho: caído
    feed(est, sim, TAG, (9.0, 0.0), [0, 1, 2, 3])
    reference = make_estimator()
    ref_sim = ZoneNodeSimulator(NODE_POSITIONS)
    feed(reference, ref_sim, TAG, (9.0, 0.0), [0, 1, 2, 3])
    reference.touch_node("N0001", at(3))
    degraded, normal = est.estimate(TAG, at(3)), reference.estimate(TAG, at(3))
    assert degraded.zone_id == normal.zone_id == "hall"
    assert degraded.degraded and not normal.degraded
    assert degraded.confidence < normal.confidence


def test_unknown_reason_is_node_outage_when_last_zone_nodes_are_down():
    est = make_estimator()
    sim = ZoneNodeSimulator(NODE_POSITIONS, sensitivity_dbm=-70)
    feed(est, sim, TAG, (1.0, 0.0), [0, 1, 2])
    # 40 s después ningún nodo del lobby ha dado señales de vida.
    result = est.estimate(TAG, at(40))
    assert result.status is EstimateStatus.UNKNOWN
    assert result.unknown_reason is UnknownReason.NODE_OUTAGE


def test_heartbeat_keeps_node_online_so_reason_is_tag_missing():
    est = make_estimator()
    sim = ZoneNodeSimulator(NODE_POSITIONS, sensitivity_dbm=-70)
    feed(est, sim, TAG, (1.0, 0.0), [0, 1, 2])
    est.touch_node("N0001", at(38))
    result = est.estimate(TAG, at(40))
    assert result.unknown_reason is UnknownReason.STALE_EVIDENCE


def test_battery_is_reported_and_flagged_when_low():
    est = make_estimator()
    est.add(obs(battery=80, t=0, seq=1))
    assert not est.estimate(TAG, at(0)).battery_low
    est.add(obs(battery=15, t=1, seq=2))
    result = est.estimate(TAG, at(1))
    assert result.battery_pct == 15 and result.battery_low


def test_tags_are_independent_and_known_tags_listed():
    est = make_estimator()
    est.add(obs(tag=TAG, node="N0001", t=0, seq=1))
    est.add(obs(tag=TAG2, node="N0004", t=0, seq=1))
    assert est.estimate(TAG, at(0)).zone_id == "lobby"
    assert est.estimate(TAG2, at(0)).zone_id == "lab"
    assert est.known_tags() == {TAG, TAG2}


def test_prune_drops_state_after_retention():
    est = make_estimator(freshness={"retention_s": 60})
    est.add(obs(t=0, seq=1))
    est.prune(at(30))
    assert TAG in est.known_tags()
    est.prune(at(61))
    assert est.known_tags() == set()
    assert est.estimate(TAG, at(61)).unknown_reason is UnknownReason.TAG_MISSING


@pytest.mark.parametrize("rssi, lo, hi", [(-55, 0.9, 1.0), (-92, 0.0, 0.5)])
def test_confidence_tracks_signal_strength(rssi, lo, hi):
    est = make_estimator()
    for node in NODE_POSITIONS:  # todos los nodos están vivos: sin penalización
        est.touch_node(node, at(3))
    for i in range(4):
        est.add(obs(rssi=rssi, t=i, seq=i + 1))
    assert lo <= est.estimate(TAG, at(3)).confidence <= hi
