"""Pruebas de validación: stale, skew, duplicados, replay e imposibles."""

import pytest
from location_support import at, make_config, obs

from location.models import RejectReason
from location.validation import ObservationValidator


@pytest.fixture
def validator():
    return ObservationValidator(make_config())


def test_valid_observation_is_accepted(validator):
    assert validator.validate(obs(t=0), at(0.2)) is None


def test_unknown_node_is_rejected(validator):
    assert validator.validate(obs(node="N9999"), at(0)) is RejectReason.UNKNOWN_NODE


@pytest.mark.parametrize("rssi", [1, 20, -128, -200])
def test_rssi_out_of_physical_range_is_impossible(validator, rssi):
    assert validator.validate(obs(rssi=rssi), at(0)) is RejectReason.IMPOSSIBLE_VALUE


@pytest.mark.parametrize("battery", [101, 200, -1])
def test_battery_out_of_range_is_impossible(validator, battery):
    result = validator.validate(obs(battery=battery), at(0))
    assert result is RejectReason.IMPOSSIBLE_VALUE


def test_unknown_battery_is_allowed(validator):
    assert validator.validate(obs(battery=None), at(0)) is None


def test_stale_observation_is_rejected_at_boundary(validator):
    # max_age_s = 10 por defecto
    assert validator.validate(obs(t=0, seq=1), at(10)) is None
    assert validator.validate(obs(t=0, seq=2), at(10.5)) is RejectReason.STALE


def test_future_timestamp_beyond_tolerance_is_clock_skew(validator):
    # future_tolerance_s = 2 por defecto
    assert validator.validate(obs(t=2, seq=1), at(0)) is None
    assert validator.validate(obs(t=3, seq=2), at(0)) is RejectReason.CLOCK_SKEW


def test_duplicate_same_tag_node_sequence(validator):
    assert validator.validate(obs(seq=5), at(0)) is None
    assert validator.validate(obs(seq=5), at(0.1)) is RejectReason.DUPLICATE


def test_replay_of_older_sequence(validator):
    assert validator.validate(obs(seq=5, t=1), at(1)) is None
    assert validator.validate(obs(seq=3, t=1.1), at(1.2)) is RejectReason.REPLAY


def test_same_packet_heard_by_two_nodes_is_not_a_duplicate(validator):
    assert validator.validate(obs(node="N0002", seq=7, rssi=-70), at(0)) is None
    assert validator.validate(obs(node="N0003", seq=7, rssi=-72), at(0)) is None


def test_sequences_are_tracked_per_tag(validator):
    assert validator.validate(obs(tag="A1B2C3D4E5F6", seq=9), at(0)) is None
    assert validator.validate(obs(tag="0A0B0C0D0E0F", seq=1), at(0)) is None


def test_sequence_wraparound_is_not_a_replay(validator):
    assert validator.validate(obs(seq=2**32 - 1, t=0), at(0)) is None
    assert validator.validate(obs(seq=0, t=1), at(1)) is None


def test_rejected_observations_do_not_poison_sequence_state(validator):
    # Stale con seq alta: no debe bloquear el siguiente paquete legítimo.
    assert validator.validate(obs(seq=100, t=0), at(60)) is RejectReason.STALE
    assert validator.validate(obs(seq=1, t=59), at(60)) is None


def test_impossible_transition_between_non_adjacent_zones(validator):
    assert validator.validate(obs(node="N0001", seq=1, t=0), at(0)) is None
    result = validator.validate(obs(node="N0004", seq=2, t=1), at(1))
    assert result is RejectReason.IMPOSSIBLE_TRANSITION


def test_adjacent_zones_are_allowed_at_the_same_time(validator):
    assert validator.validate(obs(node="N0001", seq=1, t=0), at(0)) is None
    assert validator.validate(obs(node="N0002", seq=1, t=0.1), at(0.1)) is None


def test_non_adjacent_zone_allowed_after_transit_time(validator):
    assert validator.validate(obs(node="N0001", seq=1, t=0), at(0)) is None
    assert validator.validate(obs(node="N0004", seq=2, t=6), at(6)) is None


def test_impossible_transition_does_not_consume_sequence(validator):
    assert validator.validate(obs(node="N0001", seq=1, t=0), at(0)) is None
    assert (
        validator.validate(obs(node="N0004", seq=2, t=1), at(1))
        is RejectReason.IMPOSSIBLE_TRANSITION
    )
    # El mismo paquete reintentado por el nodo válido más tarde sigue siendo nuevo.
    assert validator.validate(obs(node="N0004", seq=2, t=7), at(7)) is None


def test_prune_forgets_old_state():
    validator = ObservationValidator(make_config(freshness={"retention_s": 60}))
    validator.validate(obs(seq=1, t=0), at(0))
    assert validator.tracked_pairs == 1
    validator.prune(at(30))
    assert validator.tracked_pairs == 1
    validator.prune(at(61))
    assert validator.tracked_pairs == 0
