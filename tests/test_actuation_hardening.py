"""Pruebas de endurecimiento del adaptador: ESTOP, reinicios, secuencias y bus."""

import json

import pytest

from actuation.adapter import LinkHealth, PanTiltActuator
from actuation.config import ActuationConfig, CommsConfig
from actuation.controller import TargetObservation
from actuation.protocol import (
    Ack,
    AckStatus,
    ClearEstop,
    EmergencyStop,
    Move,
    NodeState,
    decode_command,
    encode_ack,
)
from actuation.scenarios import ClosedLoopSim
from actuation.simulator import ManualClock
from bus.hub import MetadataEnvelope, Topic

CFG = ActuationConfig()


class FlakyTransport:
    kind = "serial"

    def __init__(self):
        self.sent: list[bytes] = []
        self.inbox: list[bytes] = []
        self.fail = False

    def send(self, frame):
        if self.fail:
            raise OSError("puerto serie desconectado")
        self.sent.append(frame)

    def recv(self):
        out, self.inbox = self.inbox, []
        return out

    def close(self):
        pass

    def commands(self):
        return [decode_command(f) for f in self.sent]

    def moves(self):
        return [c for c in self.commands() if isinstance(c, Move)]

    def ack(
        self, seq, status=AckStatus.OK, state=NodeState.IDLE, pan=0.0, ms=0, last=None
    ):
        self.inbox.append(
            encode_ack(Ack(seq, status, state, pan, 0.0, ms, last_seq=last))
        )


def make(cfg=CFG, **kwargs):
    clock = ManualClock()
    tp = FlakyTransport()
    return PanTiltActuator(tp, cfg, clock, **kwargs), tp, clock


def obs(clock, ex=0.5):
    return TargetObservation(ex, 0.0, clock.now(), track_id=1)


# --- B1: el ESTOP se enclava antes de enviar ---


def test_estop_latches_before_send_even_if_transport_fails():
    act, tp, clock = make()
    tp.fail = True
    with pytest.raises(OSError):
        act.emergency_stop()
    assert act.health is LinkHealth.ESTOP
    tp.fail = False
    for _ in range(10):
        clock.advance(0.05)
        act.track(obs(clock, ex=0.9))
    assert tp.moves() == []
    assert act.health is LinkHealth.ESTOP
    # El ESTOP no entregado se reintenta en cuanto el transporte vuelve.
    assert any(isinstance(c, EmergencyStop) for c in tp.commands())


def test_estop_send_failure_is_counted_and_alarmed():
    act, tp, _clock = make()
    tp.fail = True
    with pytest.raises(OSError):
        act.emergency_stop()
    assert act.metrics()["counters"]["estop_send_failures"] == 1
    alarms = [e for t, e in act.drain_envelopes() if e.payload["kind"] == "ptz.alarm"]
    assert alarms and alarms[0].payload["reason"] == "estop_send_failed"
    assert act.metrics()["alarms"] == ["estop_send_failed"]


def test_poll_propagates_transport_errors_but_keeps_estop_latched():
    act, tp, clock = make()
    act.emergency_stop()
    tp.fail = True
    clock.advance(1.0)
    with pytest.raises(OSError):
        act.poll()
    assert act.health is LinkHealth.ESTOP


# --- N2: reinicio del nodo y vuelta del contador de milisegundos ---


def test_node_reboot_is_detected_and_telemetry_resumes():
    act, tp, clock = make()
    act.start()
    tp.ack(tp.commands()[-1].seq, pan=10.0, ms=50_000)
    act.poll()
    assert act.measured == (10.0, 0.0)
    clock.advance(0.1)
    tp.ack(tp.commands()[-1].seq, pan=9.0, ms=49_990)  # reordenado: se ignora
    act.poll()
    assert act.measured == (10.0, 0.0)
    clock.advance(0.1)
    tp.ack(tp.commands()[-1].seq, pan=0.0, ms=120)  # el nodo se reinició
    act.poll()
    assert act.measured == (0.0, 0.0)
    assert act.metrics()["counters"]["node_reboots"] == 1


def test_node_ms_uint32_wraparound_is_not_a_reboot():
    act, tp, clock = make()
    act.start()
    tp.ack(tp.commands()[-1].seq, pan=1.0, ms=2**32 - 10)
    act.poll()
    clock.advance(0.1)
    tp.ack(tp.commands()[-1].seq, pan=2.0, ms=5)
    act.poll()
    assert act.measured == (2.0, 0.0)
    assert act.metrics()["counters"]["node_reboots"] == 0


# --- N3: reinicio del host con el nodo conservando su último seq ---


def test_restarted_host_adopts_node_sequence_from_stale_ack():
    act, tp, clock = make()
    act.start()
    first = tp.commands()[-1]
    clock.advance(0.02)
    tp.ack(first.seq, status=AckStatus.STALE, ms=10, last=5000)
    act.poll()
    resent = tp.moves()[-1]
    assert resent.seq == 5001
    assert (resent.pan_deg, resent.tilt_deg) == (0.0, 0.0)
    assert act.metrics()["counters"]["seq_resyncs"] == 1


def test_stale_ack_after_sync_does_not_change_sequence():
    act, tp, clock = make()
    act.start()
    tp.ack(tp.commands()[-1].seq, ms=10)
    act.poll()
    clock.advance(0.3)
    act.track(obs(clock))
    tp.ack(tp.commands()[-1].seq, status=AckStatus.STALE, ms=20, last=40000)
    act.poll()
    clock.advance(0.3)
    act.track(obs(clock, ex=-0.9))
    assert tp.commands()[-1].seq < 1000
    assert act.metrics()["counters"]["seq_resyncs"] == 0


def test_host_restart_recovers_in_closed_loop():
    sim = ClosedLoopSim(ActuationConfig(), lambda t: (25.0, 0.0))
    sim.run(4.0)
    sim.restart_host()  # seq vuelve a 0 mientras el nodo conserva el suyo
    sim.run(6.0)
    assert abs(sim.node.position[0] - 25.0) < 2.5
    assert sim.actuator.metrics()["counters"]["seq_resyncs"] >= 1


def test_node_reboot_recovers_in_closed_loop():
    sim = ClosedLoopSim(ActuationConfig(), lambda t: (25.0, 0.0))
    sim.run(4.0)
    sim.node.reboot()  # brownout: vuelve al neutro y reinicia ms y secuencias
    sim.run(6.0)
    assert abs(sim.node.position[0] - 25.0) < 2.5
    assert sim.actuator.metrics()["counters"]["node_reboots"] >= 1


# --- N4: telemetría fuera de COMMANDS y permisos del bus ---


def test_bus_release_estop_only_from_allowlisted_source():
    act, tp, _clock = make()
    act.emergency_stop()
    n = len(tp.sent)
    release = {"kind": "ptz.release_estop"}
    act.apply_command_envelope(MetadataEnvelope(source="intruso", payload=release))
    assert len(tp.sent) == n
    assert act.metrics()["counters"]["rejected_commands"] == 1
    act.apply_command_envelope(MetadataEnvelope(source="supervisor", payload=release))
    assert isinstance(tp.commands()[-1], ClearEstop)


def test_release_allowlist_is_configurable_and_estop_open_to_anyone():
    act, tp, _clock = make(release_sources=("operador",))
    act.apply_command_envelope(
        MetadataEnvelope(source="cualquiera", payload={"kind": "ptz.estop"})
    )
    assert act.health is LinkHealth.ESTOP
    release = {"kind": "ptz.release_estop"}
    act.apply_command_envelope(MetadataEnvelope(source="supervisor", payload=release))
    assert not isinstance(tp.commands()[-1], ClearEstop)
    act.apply_command_envelope(MetadataEnvelope(source="operador", payload=release))
    assert isinstance(tp.commands()[-1], ClearEstop)


# --- N6: alarma explícita si el ESTOP no se confirma ---


def test_unconfirmed_estop_raises_explicit_alarm():
    cfg = ActuationConfig(comms=CommsConfig(estop_retries=1))
    act, _tp, clock = make(cfg)
    act.emergency_stop()
    for _ in range(4):
        clock.advance(cfg.comms.ack_timeout_s + 0.01)
        act.poll()
    assert act.metrics()["counters"]["estop_unconfirmed"] == 1
    alarms = [
        e
        for t, e in act.drain_envelopes()
        if t is Topic.EVENTS and e.payload["kind"] == "ptz.alarm"
    ]
    assert len(alarms) == 1
    assert alarms[0].payload["reason"] == "estop_unconfirmed"
    assert alarms[0].payload["severity"] == "critical"
    assert act.metrics()["alarms"] == ["estop_unconfirmed"]
    assert act.health is LinkHealth.ESTOP


# --- Metadata estricta: JSON sin NaN/inf ---


def test_every_published_envelope_is_strict_json():
    sim = ClosedLoopSim(ActuationConfig(), lambda t: (30.0, 10.0) if t < 4 else None)
    envs = []
    for _ in range(40):
        sim.run(0.25)
        envs += sim.actuator.drain_envelopes()
    sim.actuator.emergency_stop()
    sim.run(1.0)
    envs += sim.actuator.drain_envelopes()
    assert {t for t, _ in envs} >= {Topic.EVENTS, Topic.HEALTH}
    for _topic, env in envs:
        json.dumps(env.payload, allow_nan=False)


def test_undrained_envelopes_are_bounded():
    act, _tp, clock = make()
    for i in range(5000):
        clock.advance(0.3)
        act.track(obs(clock, ex=0.9 if i % 2 else -0.9))
    assert len(act.drain_envelopes()) <= 1000
    assert act.metrics()["counters"]["envelopes_dropped"] > 0


def test_sequence_adoption_repeats_if_resent_command_is_still_stale():
    act, tp, clock = make()
    act.start()
    tp.ack(tp.commands()[-1].seq, status=AckStatus.STALE, ms=10, last=100)
    act.poll()
    assert tp.moves()[-1].seq == 101
    # Un ack stale del comando anterior a la adopción no debe repetirla...
    tp.ack(1, status=AckStatus.STALE, ms=20, last=100)
    act.poll()
    assert act.metrics()["counters"]["seq_resyncs"] == 1
    # ...pero sí uno del comando reenviado (el nodo siguió con seq mayor).
    clock.advance(0.1)
    tp.ack(101, status=AckStatus.STALE, ms=30, last=300)
    act.poll()
    assert tp.moves()[-1].seq == 301
    assert act.metrics()["counters"]["seq_resyncs"] == 2
