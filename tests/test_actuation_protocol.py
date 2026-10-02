"""Pruebas del protocolo de comandos y acks del nodo Pan-Tilt (ESP32-S3)."""

import random

import pytest

from actuation.protocol import (
    Ack,
    AckStatus,
    ClearEstop,
    EmergencyStop,
    Heartbeat,
    Move,
    NodeState,
    ProtocolError,
    crc16_ccitt,
    decode_ack,
    decode_command,
    encode_ack,
    encode_command,
    seq_is_newer,
)


def _frame(body: bytes) -> bytes:
    return body + b"*" + f"{crc16_ccitt(body):04X}".encode()


def test_crc16_ccitt_false_known_vector():
    # Vector estándar CRC-16/CCITT-FALSE para "123456789".
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_move_roundtrip():
    cmd = Move(seq=42, pan_deg=12.5, tilt_deg=-3.25, speed_dps=40.0)
    frame = encode_command(cmd)
    # El framing de línea (\n) lo añade el transporte, no el codec.
    assert not frame.endswith(b"\n")
    assert decode_command(frame) == cmd


@pytest.mark.parametrize(
    "cmd",
    [
        EmergencyStop(seq=1),
        ClearEstop(seq=2),
        Heartbeat(seq=3),
        Move(seq=65535, pan_deg=-80.0, tilt_deg=45.0, speed_dps=1.0),
    ],
)
def test_all_commands_roundtrip(cmd):
    assert decode_command(encode_command(cmd)) == cmd


def test_ack_roundtrip():
    ack = Ack(
        seq=7,
        status=AckStatus.OK,
        state=NodeState.MOVING,
        pan_deg=1.5,
        tilt_deg=-2.0,
        node_ms=1234,
    )
    assert decode_ack(encode_ack(ack)) == ack


def test_corrupted_frame_is_rejected_by_crc():
    frame = bytearray(
        encode_command(Move(seq=1, pan_deg=10.0, tilt_deg=0.0, speed_dps=30.0))
    )
    frame[5] ^= 0x01
    with pytest.raises(ProtocolError, match="CRC"):
        decode_command(bytes(frame))


def test_garbage_only_raises_protocol_error():
    rng = random.Random(1234)
    for _ in range(2000):
        blob = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 80)))
        with pytest.raises(ProtocolError):
            decode_command(blob)
        with pytest.raises(ProtocolError):
            decode_ack(blob)


def test_valid_crc_but_unknown_command_is_rejected():
    with pytest.raises(ProtocolError):
        decode_command(_frame(b'{"cmd":"explode","seq":1}'))


def test_nan_or_non_numeric_fields_are_rejected():
    with pytest.raises(ProtocolError):
        decode_command(
            _frame(b'{"cmd":"move","pan":NaN,"seq":1,"speed":10.0,"tilt":0.0}')
        )
    with pytest.raises(ProtocolError):
        decode_command(
            _frame(b'{"cmd":"move","pan":"x","seq":1,"speed":10.0,"tilt":0.0}')
        )
    with pytest.raises(ProtocolError):
        decode_command(
            _frame(b'{"cmd":"move","pan":1,"seq":99999,"speed":10.0,"tilt":0.0}')
        )


def test_seq_out_of_range_is_rejected_at_construction():
    with pytest.raises(ValueError):
        Move(seq=65536, pan_deg=0.0, tilt_deg=0.0, speed_dps=10.0)
    with pytest.raises(ValueError):
        Move(seq=-1, pan_deg=0.0, tilt_deg=0.0, speed_dps=10.0)


def test_seq_is_newer_handles_wraparound():
    assert seq_is_newer(2, 1)
    assert not seq_is_newer(1, 1)
    assert not seq_is_newer(1, 2)
    assert seq_is_newer(0, 65535)
    assert seq_is_newer(5, 65530)
    assert not seq_is_newer(65530, 5)
