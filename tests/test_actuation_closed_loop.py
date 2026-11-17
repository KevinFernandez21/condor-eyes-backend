"""Pruebas de lazo cerrado: controlador + adaptador + nodo SIMULADO.

Todos los resultados son de simulación (tiempo virtual y semilla fija); no son
mediciones de hardware.
"""

import dataclasses
import itertools

import pytest

from actuation.adapter import LinkHealth
from actuation.config import ActuationConfig, SimulatorConfig
from actuation.protocol import Move, NodeState, encode_command
from actuation.scenarios import ClosedLoopSim


def cfg_with(**sim):
    base = ActuationConfig()
    return dataclasses.replace(
        base, simulator=dataclasses.replace(base.simulator, **sim)
    )


def static(pan, tilt):
    return lambda t: (pan, tilt)


def reversals(values, threshold=0.05):
    """Cuenta cambios de sentido del movimiento ignorando ruido menor al umbral."""
    count, direction, last = 0, 0, values[0]
    for v in values[1:]:
        if abs(v - last) < threshold:
            continue
        new = 1 if v > last else -1
        if direction and new != direction:
            count += 1
        direction, last = new, v
    return count


def test_converges_to_static_target_within_deadband():
    sim = ClosedLoopSim(ActuationConfig(), static(30.0, 10.0))
    sim.run(10.0)
    pan, tilt = sim.node.position
    cfg = sim.cfg
    pan_tol = cfg.control.deadband * cfg.control.hfov_deg / 2 + 0.5
    tilt_tol = cfg.control.deadband * cfg.control.vfov_deg / 2 + 0.5
    assert abs(pan - 30.0) <= pan_tol
    assert abs(tilt - 10.0) <= tilt_tol


def test_no_sustained_oscillation_with_noise_latency_and_dead_time():
    cfg = cfg_with(noise_std_deg=0.1, one_way_latency_s=0.02, dead_time_s=0.05)
    sim = ClosedLoopSim(cfg, static(20.0, -5.0), vision_latency_s=0.1, obs_noise=0.01)
    trace = sim.run(14.0)
    tail = [s for s in trace if s.t >= 8.0]
    assert reversals([s.pan for s in tail]) <= 2
    assert reversals([s.tilt for s in tail]) <= 2
    span = max(s.pan for s in tail) - min(s.pan for s in tail)
    assert span <= 2 * cfg.control.deadband * cfg.control.hfov_deg / 2
    # Sobrepaso acotado respecto al escalón de 20 grados.
    assert max(s.pan for s in trace) <= 20.0 * 1.15


def test_follows_a_slowly_moving_target():
    sim = ClosedLoopSim(
        ActuationConfig(), lambda t: (5.0 * t, 0.0), vision_latency_s=0.1
    )
    trace = sim.run(10.0)
    tail = [s for s in trace if s.t >= 5.0]
    mean_err = sum(abs(s.error_x) for s in tail) / len(tail)
    assert mean_err < 0.25


def test_lost_target_stops_then_returns_to_neutral():
    cfg = ActuationConfig()
    sim = ClosedLoopSim(cfg, lambda t: (30.0, 10.0) if t < 4.0 else None)
    trace = sim.run(14.0)
    loss_at = 4.0 + cfg.control.target_timeout_s + 0.3
    hold = [s for s in trace if loss_at <= s.t <= 4.0 + cfg.control.hold_s]
    assert hold
    assert max(s.pan for s in hold) - min(s.pan for s in hold) < 0.5
    final = trace[-1]
    assert abs(final.pan) < 0.5 and abs(final.tilt) < 0.5


def test_stale_observations_do_not_move_the_camera():
    # La visión llega con 1 s de retraso: más viejo que target_timeout_s.
    sim = ClosedLoopSim(ActuationConfig(), static(40.0, 0.0), vision_latency_s=1.0)
    sim.run(6.0)
    assert sim.node.position == pytest.approx((0.0, 0.0), abs=1e-6)


def test_lost_communication_stops_movement_safely():
    cfg = ActuationConfig()
    sim = ClosedLoopSim(cfg, lambda t: (60.0 * (t % 4) / 4, 0.0))
    sim.run(3.0)
    sim.link_up = False
    cut = sim.clock.now()
    trace = sim.run(3.0)
    pans = [s.pan for s in trace if s.t > cut]
    settle = cut + cfg.comms.node_watchdog_s + 0.15
    after = [s.pan for s in trace if s.t >= settle]
    assert max(after) - min(after) < 1e-6  # congelado en sitio
    assert sim.node.state is NodeState.FAILSAFE
    # El desplazamiento posterior al corte no supera lo que permite el watchdog.
    assert abs(pans[-1] - pans[0]) <= cfg.limits.max_speed_dps * (
        cfg.comms.node_watchdog_s + 0.15
    )
    assert sim.actuator.health is LinkHealth.LOST


def test_link_recovers_after_outage():
    sim = ClosedLoopSim(ActuationConfig(), static(25.0, 0.0))
    sim.run(3.0)
    sim.link_up = False
    sim.run(2.0)
    sim.link_up = True
    sim.run(6.0)
    assert sim.actuator.health in (LinkHealth.HEALTHY, LinkHealth.DEGRADED)
    assert abs(sim.node.position[0] - 25.0) < 2.5


def test_emergency_stop_freezes_camera_while_target_keeps_moving():
    sim = ClosedLoopSim(ActuationConfig(), lambda t: (8.0 * t, 0.0))
    sim.run(3.0)
    sim.actuator.emergency_stop()
    stop_at = sim.clock.now()
    trace = sim.run(4.0)
    frozen = [s.pan for s in trace if s.t >= stop_at + 0.15]
    assert max(frozen) - min(frozen) < 1e-6
    assert sim.node.state is NodeState.ESTOP
    assert sim.actuator.health is LinkHealth.ESTOP


def test_replayed_old_command_cannot_move_the_camera():
    sim = ClosedLoopSim(ActuationConfig(), static(10.0, 0.0))
    sim.run(4.0)
    before = sim.node.position
    replay = encode_command(Move(seq=1, pan_deg=-70.0, tilt_deg=40.0, speed_dps=60.0))
    sim.inject(replay)
    sim.run(0.3)
    assert sim.node.position == pytest.approx(before, abs=0.5)
    assert sim.node.position[0] > 5.0


def test_lossy_reordering_link_respects_limits_and_speed():
    cfg = cfg_with(jitter_s=0.05, drop_prob=0.2, noise_std_deg=0.2, seed=99)
    sim = ClosedLoopSim(cfg, lambda t: (100.0 if (t % 6) < 3 else -100.0, 70.0))
    trace = sim.run(30.0)
    lim = cfg.limits
    for s in trace:
        assert lim.pan_min_deg - 1e-9 <= s.pan <= lim.pan_max_deg + 1e-9
        assert lim.tilt_min_deg - 1e-9 <= s.tilt <= lim.tilt_max_deg + 1e-9
    step = lim.max_speed_dps * sim.physics_dt + 1e-9
    for a, b in itertools.pairwise(trace):
        assert abs(b.pan - a.pan) <= step * (b.t - a.t) / sim.physics_dt + 1e-9


def test_same_seed_gives_identical_trace():
    def run():
        cfg = cfg_with(noise_std_deg=0.2, jitter_s=0.02, drop_prob=0.1, seed=5)
        return [
            (s.t, s.pan, s.tilt)
            for s in ClosedLoopSim(cfg, static(15.0, 5.0), obs_noise=0.01).run(5.0)
        ]

    assert run() == run()


def test_metrics_are_recorded_and_labeled_simulated():
    sim = ClosedLoopSim(cfg_with(noise_std_deg=0.1), static(20.0, 0.0), obs_noise=0.01)
    sim.run(10.0)
    m = sim.actuator.metrics()
    assert m["simulated"] is True and m["transport"] == "simulated"
    assert m["latency_ms"]["samples"] > 20 and m["latency_ms"]["mean"] > 0
    assert m["tracking_error"]["mean"] is not None
    assert m["position_noise_deg"] is not None
    assert m["counters"]["moves_sent"] > 0
    assert isinstance(SimulatorConfig(), SimulatorConfig)
