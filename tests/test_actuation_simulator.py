"""Pruebas del simulador determinista del nodo Pan-Tilt (SIMULADO, no hardware)."""

import dataclasses
import itertools
import random

import pytest

from actuation.config import ActuationConfig, CommsConfig, SimulatorConfig
from actuation.protocol import (
    AckStatus,
    ClearEstop,
    EmergencyStop,
    Heartbeat,
    Move,
    NodeState,
    decode_ack,
    encode_command,
)
from actuation.simulator import ManualClock, SimulatedPanTiltNode, SimulatedTransport


def make_rig(watchdog_s=100.0, **sim_overrides):
    base = {
        "one_way_latency_s": 0.01,
        "jitter_s": 0.0,
        "dead_time_s": 0.03,
        "noise_std_deg": 0.0,
        "drop_prob": 0.0,
        "servo_speed_dps": 120.0,
        "seed": 7,
        "physics_dt_s": 0.002,
    }
    base.update(sim_overrides)
    cfg = ActuationConfig(
        comms=CommsConfig(node_watchdog_s=watchdog_s),
        simulator=SimulatorConfig(**base),
    )
    clock = ManualClock()
    node = SimulatedPanTiltNode(cfg, clock)
    link = SimulatedTransport(node, clock, cfg.simulator)
    return cfg, clock, node, link


def run(clock, link, seconds, dt=0.005, on_step=None):
    """Avanza el reloj; devuelve los acks decodificados recibidos."""
    acks = []
    steps = round(seconds / dt)
    for _ in range(steps):
        clock.advance(dt)
        for frame in link.recv():
            acks.append(decode_ack(frame))
        if on_step:
            on_step()
    return acks


def test_node_starts_at_neutral_and_idle():
    cfg, _clock, node, _link = make_rig()
    assert node.position == (cfg.limits.neutral_pan_deg, cfg.limits.neutral_tilt_deg)
    assert node.state is NodeState.IDLE


def test_move_reaches_target_respecting_speed_and_dead_time():
    _cfg, clock, node, link = make_rig()
    link.send(encode_command(Move(seq=1, pan_deg=30.0, tilt_deg=10.0, speed_dps=40.0)))
    positions = [node.position]
    acks = run(clock, link, 2.0, on_step=lambda: positions.append(node.position))
    assert acks[0].status is AckStatus.OK
    assert node.position == pytest.approx((30.0, 10.0), abs=1e-6)
    # Tiempo muerto + latencia: el primer movimiento no es instantáneo.
    first_move = next(i for i, p in enumerate(positions) if p != positions[0])
    assert first_move * 0.005 >= 0.03
    # Velocidad por eje nunca por encima de la pedida (40 dps).
    for a, b in itertools.pairwise(positions):
        assert abs(b[0] - a[0]) <= 40.0 * 0.005 + 1e-9
        assert abs(b[1] - a[1]) <= 40.0 * 0.005 + 1e-9


def test_ack_arrives_after_round_trip_latency():
    _cfg, clock, _node, link = make_rig(one_way_latency_s=0.05)
    link.send(encode_command(Heartbeat(seq=1)))
    clock.advance(0.09)
    assert link.recv() == []
    clock.advance(0.02)
    frames = link.recv()
    assert len(frames) == 1
    assert decode_ack(frames[0]).seq == 1


def test_command_beyond_mechanical_limits_is_clamped():
    cfg, clock, node, link = make_rig()
    link.send(
        encode_command(Move(seq=1, pan_deg=500.0, tilt_deg=-500.0, speed_dps=40.0))
    )
    run(clock, link, 5.0)
    assert node.position == pytest.approx(
        (cfg.limits.pan_max_deg, cfg.limits.tilt_min_deg), abs=1e-6
    )


def test_speed_above_hard_limit_is_clamped():
    cfg, clock, node, link = make_rig()
    link.send(
        encode_command(Move(seq=1, pan_deg=80.0, tilt_deg=0.0, speed_dps=10000.0))
    )
    prev = [node.position]
    peak = [0.0]

    def watch():
        peak[0] = max(peak[0], abs(node.position[0] - prev[0][0]) / 0.005)
        prev[0] = node.position

    run(clock, link, 3.0, on_step=watch)
    assert peak[0] <= cfg.limits.max_speed_dps + 1e-6


def test_duplicate_command_is_acked_without_moving():
    _cfg, clock, node, link = make_rig()
    link.send(encode_command(Move(seq=5, pan_deg=10.0, tilt_deg=0.0, speed_dps=60.0)))
    run(clock, link, 1.0)
    assert node.position[0] == pytest.approx(10.0)
    link.send(encode_command(Move(seq=5, pan_deg=-50.0, tilt_deg=0.0, speed_dps=60.0)))
    acks = run(clock, link, 1.0)
    assert acks[0].status is AckStatus.DUPLICATE
    assert node.position[0] == pytest.approx(10.0)


def test_stale_command_is_ignored_without_moving():
    _cfg, clock, node, link = make_rig()
    link.send(encode_command(Move(seq=10, pan_deg=10.0, tilt_deg=0.0, speed_dps=60.0)))
    run(clock, link, 1.0)
    link.send(encode_command(Move(seq=9, pan_deg=-70.0, tilt_deg=0.0, speed_dps=60.0)))
    acks = run(clock, link, 1.0)
    assert acks[0].status is AckStatus.STALE
    assert node.position[0] == pytest.approx(10.0)


def test_sequence_wraparound_is_accepted():
    _cfg, clock, node, link = make_rig()
    link.send(
        encode_command(Move(seq=65535, pan_deg=5.0, tilt_deg=0.0, speed_dps=60.0))
    )
    run(clock, link, 0.5)
    link.send(encode_command(Move(seq=0, pan_deg=15.0, tilt_deg=0.0, speed_dps=60.0)))
    acks = run(clock, link, 1.0)
    assert acks[0].status is AckStatus.OK
    assert node.position[0] == pytest.approx(15.0)


def test_corrupted_frame_gets_no_ack_and_no_motion():
    _cfg, clock, node, link = make_rig()
    frame = bytearray(
        encode_command(Move(seq=1, pan_deg=30.0, tilt_deg=0.0, speed_dps=60.0))
    )
    frame[8] ^= 0x04
    link.send(bytes(frame))
    acks = run(clock, link, 1.0)
    assert acks == []
    assert node.position == (0.0, 0.0)
    assert node.crc_errors == 1


def test_emergency_stop_freezes_and_latches():
    _cfg, clock, node, link = make_rig()
    link.send(encode_command(Move(seq=1, pan_deg=70.0, tilt_deg=0.0, speed_dps=30.0)))
    run(clock, link, 0.5)
    mid = node.position[0]
    assert 0.0 < mid < 70.0
    link.send(encode_command(EmergencyStop(seq=2)))
    acks = run(clock, link, 0.2)
    assert acks[0].state is NodeState.ESTOP
    frozen = node.position[0]
    run(clock, link, 1.0)
    assert node.position[0] == pytest.approx(frozen)
    # Con ESTOP activo los movimientos se rechazan.
    link.send(encode_command(Move(seq=3, pan_deg=-60.0, tilt_deg=0.0, speed_dps=60.0)))
    acks = run(clock, link, 1.0)
    assert acks[0].status is AckStatus.ESTOP
    assert node.position[0] == pytest.approx(frozen)
    # Solo un clear más nuevo libera; luego se puede mover.
    link.send(encode_command(ClearEstop(seq=4)))
    run(clock, link, 0.2)
    assert node.state is not NodeState.ESTOP
    link.send(encode_command(Move(seq=5, pan_deg=-20.0, tilt_deg=0.0, speed_dps=60.0)))
    run(clock, link, 2.0)
    assert node.position[0] == pytest.approx(-20.0)


def test_estop_is_honored_even_with_stale_sequence():
    _cfg, clock, node, link = make_rig()
    link.send(encode_command(Move(seq=100, pan_deg=70.0, tilt_deg=0.0, speed_dps=30.0)))
    run(clock, link, 0.4)
    link.send(encode_command(EmergencyStop(seq=3)))  # seq viejo
    run(clock, link, 0.2)
    assert node.state is NodeState.ESTOP


def test_stale_clear_does_not_release_estop():
    _cfg, clock, node, link = make_rig()
    link.send(encode_command(EmergencyStop(seq=50)))
    run(clock, link, 0.2)
    link.send(encode_command(ClearEstop(seq=40)))
    run(clock, link, 0.2)
    assert node.state is NodeState.ESTOP


def test_comms_watchdog_freezes_the_node_in_place():
    cfg, clock, node, link = make_rig(watchdog_s=0.75)
    link.send(encode_command(Move(seq=1, pan_deg=70.0, tilt_deg=0.0, speed_dps=20.0)))
    run(clock, link, 0.3)  # el host "muere" aquí
    run(clock, link, cfg.comms.node_watchdog_s + 0.5)
    assert node.state is NodeState.FAILSAFE
    frozen = node.position[0]
    assert 0.0 < frozen < 70.0
    run(clock, link, 2.0)
    assert node.position[0] == pytest.approx(frozen)
    # Al volver el enlace, un comando válido nuevo recupera el control.
    link.send(encode_command(Move(seq=2, pan_deg=0.0, tilt_deg=0.0, speed_dps=60.0)))
    run(clock, link, 0.5)
    assert node.state is NodeState.IDLE
    assert node.position[0] == pytest.approx(0.0)


def test_noise_is_applied_to_reported_position_only():
    _cfg, clock, node, link = make_rig(noise_std_deg=0.5)
    reported = []
    for seq in range(1, 60):
        link.send(encode_command(Heartbeat(seq=seq)))
        reported += [a.pan_deg for a in run(clock, link, 0.05)]
    assert node.position == (0.0, 0.0)
    assert len(set(reported)) > 10
    assert max(abs(x) for x in reported) < 4.0


def test_simulation_is_deterministic_for_a_seed():
    def trace(seed):
        _cfg, clock, _node, link = make_rig(
            noise_std_deg=0.3, jitter_s=0.01, drop_prob=0.1, seed=seed
        )
        out = []
        for seq in range(1, 40):
            link.send(
                encode_command(
                    Move(seq=seq, pan_deg=seq % 7 * 5.0, tilt_deg=0.0, speed_dps=50.0)
                )
            )
            out += [(a.seq, a.pan_deg) for a in run(clock, link, 0.05)]
        return out

    assert trace(11) == trace(11)
    assert trace(11) != trace(12)


class _Watcher:
    """Registra violaciones de límites mecánicos o de velocidad entre pasos."""

    def __init__(self, node, lim):
        self.node, self.lim = node, lim
        self.prev = node.position
        self.violations = []

    def __call__(self):
        p, q, lim = self.node.position, self.prev, self.lim
        if not (lim.pan_min_deg - 1e-9 <= p[0] <= lim.pan_max_deg + 1e-9):
            self.violations.append(("pan", p))
        if not (lim.tilt_min_deg - 1e-9 <= p[1] <= lim.tilt_max_deg + 1e-9):
            self.violations.append(("tilt", p))
        step = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
        if step > lim.max_speed_dps * 0.005 + 1e-9:
            self.violations.append(("speed", q, p))
        self.prev = p


def test_random_command_streams_never_violate_limits_or_speed():
    """Propiedad: ante cualquier flujo de comandos el nodo respeta límites y velocidad."""
    for case in range(60):
        rng = random.Random(case)
        cfg, clock, node, link = make_rig(
            jitter_s=rng.choice([0.0, 0.02]),
            drop_prob=rng.choice([0.0, 0.2]),
            seed=case,
        )
        watch = _Watcher(node, cfg.limits)

        for _ in range(60):
            kind = rng.random()
            seq = rng.randrange(65536)
            if kind < 0.75:
                cmd = Move(
                    seq=seq,
                    pan_deg=rng.uniform(-1000, 1000),
                    tilt_deg=rng.uniform(-1000, 1000),
                    speed_dps=rng.choice([0.1, 10, 60, 5000, 1e9]),
                )
            elif kind < 0.85:
                cmd = EmergencyStop(seq=seq)
            elif kind < 0.9:
                cmd = ClearEstop(seq=seq)
            else:
                cmd = Heartbeat(seq=seq)
            link.send(encode_command(cmd))
            run(clock, link, rng.uniform(0.01, 0.2), on_step=watch)
        assert watch.violations == []


def test_simulator_config_is_labeled_simulated():
    cfg, _clock, _node, link = make_rig()
    assert link.kind == "simulated"
    assert dataclasses.is_dataclass(cfg.simulator)
