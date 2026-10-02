"""Pruebas del controlador error de imagen -> comando Pan-Tilt acotado."""

import math
import random

import pytest

from actuation.config import ActuationConfig, ControlConfig, LimitsConfig
from actuation.controller import (
    ControlState,
    TargetObservation,
    TrackingController,
)

DT = 0.05


def make(**control):
    cfg = ActuationConfig(control=ControlConfig(**control))
    return cfg, TrackingController(cfg)


def obs(ex, ey, t):
    return TargetObservation(error_x=ex, error_y=ey, timestamp=t, track_id=1)


def test_target_inside_deadband_produces_no_motion():
    _cfg, ctl = make(deadband=0.05)
    for i in range(20):
        d = ctl.update(obs(0.03, -0.02, i * DT), i * DT)
        assert (d.pan_deg, d.tilt_deg) == (0.0, 0.0)
        assert d.state is ControlState.TRACKING


def test_target_right_pans_positive_and_below_tilts_negative():
    _cfg, ctl = make(smoothing_alpha=1.0)
    d = ctl.update(obs(0.5, 0.5, 0.0), 0.0)
    assert d.pan_deg > 0.0
    assert d.tilt_deg < 0.0


def test_step_is_rate_limited_per_cycle():
    _cfg, ctl = make(smoothing_alpha=1.0, max_speed_dps=30.0)
    prev = ctl.update(obs(1.0, 0.0, 0.0), 0.0).pan_deg
    assert prev <= 30.0 * DT + 1e-9
    for i in range(1, 30):
        d = ctl.update(obs(1.0, 0.0, i * DT), i * DT)
        assert d.pan_deg - prev <= 30.0 * DT + 1e-9
        assert d.speed_dps <= 30.0
        prev = d.pan_deg


def test_persistent_error_saturates_at_mechanical_limits():
    cfg, ctl = make(smoothing_alpha=1.0)
    d = None
    for i in range(400):
        d = ctl.update(obs(1.0, -1.0, i * DT), i * DT)
    assert d.pan_deg == cfg.limits.pan_max_deg
    assert d.tilt_deg == cfg.limits.tilt_max_deg


def test_smoothing_attenuates_a_single_outlier():
    _c1, raw = make(smoothing_alpha=1.0, max_speed_dps=60.0)
    _c2, smooth = make(smoothing_alpha=0.2, max_speed_dps=60.0)
    for ctl in (raw, smooth):
        ctl.update(obs(0.0, 0.0, 0.0), 0.0)
    a = raw.update(obs(0.9, 0.0, 0.5), 0.5)
    b = smooth.update(obs(0.9, 0.0, 0.5), 0.5)
    assert b.pan_deg < a.pan_deg


def test_lost_target_freezes_then_returns_to_neutral_after_hold():
    cfg, ctl = make(smoothing_alpha=1.0, hold_s=0.5, neutral_speed_dps=20.0)
    t = 0.0
    for _ in range(10):
        d = ctl.update(obs(0.8, 0.8, t), t)
        t += DT
    pan_at_loss, tilt_at_loss = d.pan_deg, d.tilt_deg
    assert pan_at_loss > 0
    d = ctl.update(None, t)
    assert d.state is ControlState.HOLDING
    assert (d.pan_deg, d.tilt_deg) == (pan_at_loss, tilt_at_loss)
    # Dentro del periodo de espera no se mueve.
    for _ in range(5):
        t += DT
        d = ctl.update(None, t)
        assert (d.pan_deg, d.tilt_deg) == (pan_at_loss, tilt_at_loss)
    # Pasado hold_s regresa al neutro con velocidad limitada.
    states = set()
    prev = d.pan_deg
    for _ in range(600):
        t += DT
        d = ctl.update(None, t)
        states.add(d.state)
        assert abs(d.pan_deg - prev) <= 20.0 * DT + 1e-9
        if d.state is not ControlState.HOLDING:
            assert d.speed_dps <= 20.0
        prev = d.pan_deg
    assert ControlState.RETURNING in states
    assert d.state is ControlState.NEUTRAL
    assert (d.pan_deg, d.tilt_deg) == (
        cfg.limits.neutral_pan_deg,
        cfg.limits.neutral_tilt_deg,
    )


def test_lost_target_without_return_to_neutral_holds_forever():
    _cfg, ctl = make(smoothing_alpha=1.0, hold_s=0.2, return_to_neutral=False)
    t = 0.0
    for _ in range(10):
        d = ctl.update(obs(0.8, 0.0, t), t)
        t += DT
    held = d.pan_deg
    for _ in range(200):
        t += DT
        d = ctl.update(None, t)
    assert d.state is ControlState.HOLDING
    assert d.pan_deg == held


def test_loss_freezes_at_measured_position_when_available():
    _cfg, ctl = make(smoothing_alpha=1.0)
    t = 0.0
    for _ in range(10):
        d = ctl.update(obs(1.0, 0.0, t), t)
        t += DT
    commanded = d.pan_deg
    d = ctl.update(None, t, measured=(commanded - 3.0, 0.0))
    assert d.pan_deg == pytest.approx(commanded - 3.0)


def test_stale_observation_is_treated_as_lost():
    _cfg, ctl = make(target_timeout_s=0.3)
    ctl.update(obs(0.5, 0.0, 0.0), 0.0)
    d = ctl.update(obs(0.5, 0.0, 0.0), 1.0)  # observación de hace 1 s
    assert d.state is ControlState.HOLDING


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_observation_is_treated_as_lost(bad):
    _cfg, ctl = make()
    d = ctl.update(obs(bad, 0.0, 0.0), 0.0)
    assert d.state is ControlState.HOLDING
    d = ctl.update(obs(0.0, bad, DT), DT)
    assert d.state is ControlState.HOLDING


def test_reacquiring_target_resumes_tracking():
    _cfg, ctl = make(smoothing_alpha=1.0, hold_s=0.1)
    t = 0.0
    for _ in range(5):
        ctl.update(obs(0.8, 0.0, t), t)
        t += DT
    for _ in range(20):
        ctl.update(None, t)
        t += DT
    d = ctl.update(obs(0.8, 0.0, t), t)
    assert d.state is ControlState.TRACKING


def test_tracking_error_is_recorded():
    _cfg, ctl = make(smoothing_alpha=1.0)
    ctl.update(obs(0.3, 0.4, 0.0), 0.0)
    assert ctl.last_error == pytest.approx(0.5)
    ctl.update(None, DT)
    assert ctl.last_error is None


def test_property_decisions_never_exceed_limits_or_rate():
    """Propiedad: con entradas aleatorias (incluso basura) la salida siempre es segura."""
    for case in range(300):
        rng = random.Random(case)
        limits = LimitsConfig(
            pan_min_deg=-rng.uniform(10, 90),
            pan_max_deg=rng.uniform(10, 90),
            tilt_min_deg=-rng.uniform(10, 40),
            tilt_max_deg=rng.uniform(10, 60),
            max_speed_dps=rng.uniform(20, 120),
        )
        control = ControlConfig(
            kp=rng.uniform(0.1, 1.0),
            deadband=rng.uniform(0.0, 0.3),
            smoothing_alpha=rng.uniform(0.05, 1.0),
            max_speed_dps=rng.uniform(5, limits.max_speed_dps),
            neutral_speed_dps=rng.uniform(5, limits.max_speed_dps),
            hold_s=rng.uniform(0, 2),
            return_to_neutral=rng.random() < 0.7,
        )
        cfg = ActuationConfig(limits=limits, control=control)
        ctl = TrackingController(cfg)
        t = 0.0
        prev = (limits.neutral_pan_deg, limits.neutral_tilt_deg)
        prev_t = None
        for _ in range(200):
            t += rng.choice([0.0, 0.01, 0.05, 0.05, 0.2, 3.0])
            kind = rng.random()
            if kind < 0.2:
                o = None
            elif kind < 0.3:
                o = obs(rng.choice([math.nan, math.inf]), rng.uniform(-1, 1), t)
            else:
                o = obs(
                    rng.uniform(-5, 5), rng.uniform(-5, 5), t - rng.choice([0, 0, 0.9])
                )
            measured = None
            if rng.random() < 0.5:
                measured = (
                    rng.uniform(limits.pan_min_deg, limits.pan_max_deg),
                    rng.uniform(limits.tilt_min_deg, limits.tilt_max_deg),
                )
            d = ctl.update(o, t, measured=measured)
            assert limits.pan_min_deg <= d.pan_deg <= limits.pan_max_deg
            assert limits.tilt_min_deg <= d.tilt_deg <= limits.tilt_max_deg
            assert 0 < d.speed_dps <= limits.max_speed_dps
            assert math.isfinite(d.pan_deg) and math.isfinite(d.tilt_deg)
            if d.state in (ControlState.TRACKING, ControlState.RETURNING):
                dt = (
                    1 / control.control_rate_hz
                    if prev_t is None
                    else min(max(t - prev_t, 0.0), 0.5)
                )
                max_step = d.speed_dps * dt + 1e-9
                assert abs(d.pan_deg - prev[0]) <= max_step
                assert abs(d.tilt_deg - prev[1]) <= max_step
            prev, prev_t = (d.pan_deg, d.tilt_deg), t
