"""Pruebas del adaptador PanTiltActuator con un transporte falso (sin hardware)."""

import json
import sys
import types

import pytest

from actuation.adapter import LinkHealth, PanTiltActuator
from actuation.config import ActuationConfig
from actuation.controller import ControlDecision, ControlState, TargetObservation
from actuation.protocol import (
    Ack,
    AckStatus,
    ClearEstop,
    EmergencyStop,
    Heartbeat,
    Move,
    NodeState,
    decode_command,
    encode_ack,
)
from actuation.simulator import ManualClock
from actuation.transport import SerialTransport, TransportError
from bus.hub import Topic

CFG = ActuationConfig()


class FakeTransport:
    kind = "serial"

    def __init__(self):
        self.sent: list[bytes] = []
        self.inbox: list[bytes] = []
        self.closed = False

    def send(self, frame):
        self.sent.append(frame)

    def recv(self):
        out, self.inbox = self.inbox, []
        return out

    def close(self):
        self.closed = True

    def commands(self):
        return [decode_command(f) for f in self.sent]

    def moves(self):
        return [c for c in self.commands() if isinstance(c, Move)]

    def ack(
        self, seq, status=AckStatus.OK, state=NodeState.IDLE, pan=0.0, tilt=0.0, ms=0
    ):
        self.inbox.append(encode_ack(Ack(seq, status, state, pan, tilt, ms)))


def make(cfg=CFG, controller=None):
    clock = ManualClock()
    tp = FakeTransport()
    act = PanTiltActuator(tp, cfg, clock, camera_id="cam-01", controller=controller)
    return act, tp, clock


def obs(clock, ex=0.5, ey=0.0):
    return TargetObservation(ex, ey, clock.now(), track_id=3)


def test_start_sends_neutral_move_at_safe_speed():
    act, tp, _clock = make()
    act.start()
    (move,) = tp.moves()
    assert (move.pan_deg, move.tilt_deg) == (0.0, 0.0)
    assert move.speed_dps == CFG.control.neutral_speed_dps


def test_sequence_numbers_increase_and_wrap():
    act, tp, clock = make()
    act._seq = 65534
    for _ in range(4):
        clock.advance(CFG.comms.heartbeat_interval_s + 0.01)
        act.poll()
    seqs = [c.seq for c in tp.commands()]
    assert seqs == [65535, 0, 1, 2]


def test_track_sends_bounded_move_and_clamps_even_if_controller_misbehaves():
    class Rogue:
        last_error = None

        def update(self, observation, now, measured=None):
            return ControlDecision(1e6, -1e6, 1e9, ControlState.TRACKING)

        def resync(self, position):
            pass

    act, tp, clock = make(controller=Rogue())
    act.track(obs(clock))
    (move,) = tp.moves()
    assert move.pan_deg == CFG.limits.pan_max_deg
    assert move.tilt_deg == CFG.limits.tilt_min_deg
    assert move.speed_dps == CFG.limits.max_speed_dps


def test_unchanged_decision_is_not_resent_but_heartbeat_keeps_node_alive():
    act, tp, clock = make()
    act.track(obs(clock, ex=0.0))
    n = len(tp.sent)
    clock.advance(0.05)
    act.track(obs(clock, ex=0.0))
    assert len(tp.sent) == n  # sin cambios: nada que enviar
    clock.advance(CFG.comms.heartbeat_interval_s)
    act.track(obs(clock, ex=0.0))
    assert isinstance(tp.commands()[-1], Heartbeat)


def test_ack_updates_latency_and_measured_position():
    act, tp, clock = make()
    act.start()
    clock.advance(0.04)
    tp.ack(tp.commands()[0].seq, pan=1.5, tilt=-0.5, ms=40)
    act.poll()
    m = act.metrics()
    assert m["latency_ms"]["last"] == pytest.approx(40.0)
    assert act.measured == (1.5, -0.5)


def test_out_of_order_ack_does_not_rewind_measured_position():
    act, tp, clock = make()
    act.track(obs(clock))
    clock.advance(0.3)
    act.track(obs(clock))
    s1, s2 = (m.seq for m in tp.moves()[-2:])
    tp.ack(s2, pan=10.0, ms=300)
    act.poll()
    tp.ack(s1, pan=2.0, ms=100)  # ack viejo llegando tarde
    act.poll()
    assert act.measured[0] == 10.0


def test_corrupted_ack_is_counted_and_ignored():
    act, tp, _clock = make()
    act.start()
    tp.inbox.append(b'{"ack":1}*0000')
    act.poll()
    assert act.metrics()["counters"]["bad_frames"] == 1


def test_lost_link_stops_issuing_moves_and_recovers_on_ack():
    act, tp, clock = make()
    act.track(obs(clock))  # sin acks de ahí en adelante
    t_end = CFG.comms.comms_timeout_s + 0.1
    sent_moves = len(tp.moves())
    while clock.now() < t_end:
        clock.advance(0.05)
        act.track(obs(clock, ex=0.9))
    assert act.health is LinkHealth.LOST
    moves_after_loss = len(tp.moves())
    for _ in range(20):
        clock.advance(0.05)
        act.track(obs(clock, ex=0.9))
    assert len(tp.moves()) == moves_after_loss  # ya no se emiten movimientos
    assert moves_after_loss >= sent_moves
    # Un ack vuelve a poner el enlace en servicio.
    tp.ack(tp.commands()[-1].seq, ms=1)
    act.poll()
    assert act.health in (LinkHealth.HEALTHY, LinkHealth.DEGRADED)


def test_ack_timeouts_are_counted():
    act, _tp, clock = make()
    act.track(obs(clock))
    clock.advance(CFG.comms.ack_timeout_s + 0.01)
    act.poll()
    assert act.metrics()["counters"]["ack_timeouts"] >= 1


def test_emergency_stop_blocks_moves_retries_and_needs_release():
    act, tp, clock = make()
    act.track(obs(clock))
    act.emergency_stop()
    assert act.health is LinkHealth.ESTOP
    assert isinstance(tp.commands()[-1], EmergencyStop)
    estop_seq = tp.commands()[-1].seq
    before = len(tp.moves())
    for _ in range(5):
        clock.advance(0.05)
        act.track(obs(clock, ex=0.9))
    assert len(tp.moves()) == before
    # Sin ack, se reintenta con el mismo seq.
    clock.advance(CFG.comms.ack_timeout_s + 0.01)
    act.poll()
    resent = [c for c in tp.commands() if isinstance(c, EmergencyStop)]
    assert len(resent) >= 2
    assert {c.seq for c in resent} == {estop_seq}
    # Confirmado por el nodo, sigue bloqueado hasta release_estop().
    tp.ack(estop_seq, state=NodeState.ESTOP, ms=10)
    act.poll()
    act.track(obs(clock, ex=0.9))
    assert len(tp.moves()) == before
    act.release_estop()
    clear = tp.commands()[-1]
    assert isinstance(clear, ClearEstop) and clear.seq != estop_seq
    tp.ack(clear.seq, state=NodeState.IDLE, ms=20)
    act.poll()
    assert act.health is not LinkHealth.ESTOP
    clock.advance(0.05)
    act.track(obs(clock, ex=-0.9))
    assert len(tp.moves()) == before + 1


def test_node_reporting_estop_latches_host_health():
    act, tp, _clock = make()
    act.start()
    tp.ack(tp.commands()[0].seq, status=AckStatus.ESTOP, state=NodeState.ESTOP, ms=5)
    act.poll()
    assert act.health is LinkHealth.ESTOP


def test_envelopes_carry_only_serializable_metadata_and_label_simulation():
    act, tp, clock = make()
    act.track(obs(clock))
    tp.ack(tp.commands()[-1].seq, ms=30)
    clock.advance(CFG.comms.health_report_interval_s + 0.01)
    act.poll()
    envs = act.drain_envelopes()
    topics = {topic for topic, _ in envs}
    assert Topic.COMMANDS in topics and Topic.HEALTH in topics
    for _topic, env in envs:
        assert env.source == "actuation" and env.stream_id == "cam-01"
        json.dumps(env.payload)  # serializable, sin frames
        assert env.payload["simulated"] is False
    assert act.drain_envelopes() == []


def test_envelope_payload_is_marked_simulated_for_simulated_transport():
    class Sim(FakeTransport):
        kind = "simulated"

    clock = ManualClock()
    act = PanTiltActuator(Sim(), CFG, clock)
    act.start()
    (_t, env), *_ = act.drain_envelopes()
    assert env.payload["simulated"] is True


def test_estop_request_via_bus_envelope():
    from bus.hub import MetadataEnvelope

    act, tp, _clock = make()
    act.apply_command_envelope(
        MetadataEnvelope(source="supervisor", payload={"kind": "ptz.estop"})
    )
    assert act.health is LinkHealth.ESTOP
    assert isinstance(tp.commands()[-1], EmergencyStop)
    # Payload desconocido: se ignora sin efecto.
    n = len(tp.sent)
    act.apply_command_envelope(MetadataEnvelope(source="x", payload={"kind": "otra"}))
    assert len(tp.sent) == n


def test_close_closes_transport():
    act, tp, _clock = make()
    act.close()
    assert tp.closed


def test_serial_transport_without_pyserial_fails_with_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)
    with pytest.raises(TransportError, match="pyserial"):
        SerialTransport("COM3")


def test_serial_transport_frames_lines(monkeypatch):
    class FakePort:
        def __init__(self, *a, **k):
            self.buf = b'abc\r\n{"x":1}*AAAA\npartial'
            self.written = b""

        @property
        def in_waiting(self):
            return len(self.buf)

        def read(self, n):
            out, self.buf = self.buf[:n], self.buf[n:]
            return out

        def write(self, data):
            self.written += data

        def close(self):
            pass

    fake = types.ModuleType("serial")
    fake.Serial = FakePort
    monkeypatch.setitem(sys.modules, "serial", fake)
    tp = SerialTransport("COM3")
    assert tp.recv() == [b"abc", b'{"x":1}*AAAA']
    tp.send(b"hola")
    assert tp._port.written == b"hola\n"
