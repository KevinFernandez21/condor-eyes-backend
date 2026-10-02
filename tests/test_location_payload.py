"""Pruebas del contrato de payload entre nodos de zona y localización."""

import json
from datetime import UTC, datetime

import pytest

from location.models import Channel, TagObservation
from location.payload import (
    BINARY_SIZE,
    PayloadError,
    crc16_ccitt,
    encode_binary,
    encode_json,
    parse_binary,
    parse_json,
)

T0 = datetime(2026, 1, 1, 12, 0, 0, 250_000, tzinfo=UTC)


def make_obs(**overrides):
    data = {
        "tag_id": "A1B2C3D4E5F6",
        "node_id": "N0007",
        "rssi_dbm": -67,
        "timestamp": T0,
        "sequence": 1234,
        "battery_pct": 87,
        "channel": Channel.BLE,
    }
    data.update(overrides)
    return TagObservation(**data)


def valid_json(**overrides):
    data = {
        "v": 1,
        "ch": "ble",
        "tag": "a1b2c3d4e5f6",
        "node": "N0007",
        "rssi": -67,
        "ts": int(T0.timestamp() * 1000),
        "seq": 1234,
        "bat": 87,
    }
    data.update(overrides)
    return data


def test_parse_json_normalizes_tag_and_timestamp():
    obs = parse_json(json.dumps(valid_json()))
    assert obs == make_obs()
    assert obs.tag_id == "A1B2C3D4E5F6"
    assert obs.timestamp.tzinfo is not None


def test_parse_json_accepts_bytes_and_missing_battery():
    raw = valid_json()
    del raw["bat"]
    obs = parse_json(json.dumps(raw).encode())
    assert obs.battery_pct is None


@pytest.mark.parametrize(
    "mutation",
    [
        {"v": 2},
        {"tag": "XYZ"},
        {"tag": "A1B2C3"},
        {"node": ""},
        {"node": "con espacios"},
        {"rssi": "fuerte"},
        {"rssi": True},
        {"ts": "ayer"},
        {"seq": -1},
        {"seq": 2**32},
        {"ch": "wifi"},
        {"bat": "lleno"},
    ],
)
def test_parse_json_rejects_malformed_fields(mutation):
    with pytest.raises(PayloadError):
        parse_json(json.dumps(valid_json(**mutation)))


@pytest.mark.parametrize("missing", ["v", "tag", "node", "rssi", "ts", "seq"])
def test_parse_json_rejects_missing_required_fields(missing):
    raw = valid_json()
    del raw[missing]
    with pytest.raises(PayloadError, match=missing):
        parse_json(json.dumps(raw))


def test_parse_json_rejects_non_object_and_garbage():
    for raw in ("[1,2]", "no es json", ""):
        with pytest.raises(PayloadError):
            parse_json(raw)


def test_json_roundtrip():
    obs = make_obs()
    assert parse_json(encode_json(obs)) == obs


def test_binary_roundtrip_and_size():
    obs = make_obs(channel=Channel.LORA)
    raw = encode_binary(obs)
    assert len(raw) == BINARY_SIZE == 26
    assert parse_binary(raw) == obs


def test_binary_unknown_battery_is_none():
    obs = make_obs(battery_pct=None)
    assert parse_binary(encode_binary(obs)).battery_pct is None


def test_binary_detects_corruption_with_crc():
    raw = bytearray(encode_binary(make_obs()))
    raw[10] ^= 0xFF
    with pytest.raises(PayloadError, match="CRC"):
        parse_binary(bytes(raw))


def test_binary_rejects_wrong_length_and_version():
    with pytest.raises(PayloadError, match="longitud"):
        parse_binary(b"\x01\x02")
    raw = bytearray(encode_binary(make_obs()))
    raw[0] = 9
    raw[-2:] = crc16_ccitt(bytes(raw[:-2])).to_bytes(2, "little")
    with pytest.raises(PayloadError, match="versión"):
        parse_binary(bytes(raw))


def test_crc16_ccitt_known_vector():
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_binary_encode_requires_numeric_node_id():
    with pytest.raises(ValueError, match="nodo"):
        encode_binary(make_obs(node_id="zona-lobby"))


def test_repr_does_not_leak_tag_id():
    assert "A1B2C3D4E5F6" not in repr(make_obs())


# --- entradas hostiles -------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["[" * 100_000, '{"a":' * 100_000, b"[" * 100_000, "x" * 100_000],
    ids=["corchetes", "objetos-anidados", "corchetes-bytes", "sobredimensionado"],
)
def test_parse_json_rejects_hostile_or_oversized_input(raw):
    with pytest.raises(PayloadError):
        parse_json(raw)


@pytest.mark.parametrize("raw", [None, 123, 1.5, [], {}, object()])
def test_parse_json_rejects_wrong_input_type(raw):
    with pytest.raises(PayloadError, match="tipo"):
        parse_json(raw)


def test_parse_json_rejects_invalid_utf8():
    with pytest.raises(PayloadError):
        parse_json(b"\xff\xfe{")


@pytest.mark.parametrize("raw", ["x" * 26, None, 26, [0] * 26])
def test_parse_binary_rejects_wrong_input_type(raw):
    with pytest.raises(PayloadError, match="tipo"):
        parse_binary(raw)


def test_parse_json_rejects_non_finite_numbers():
    for bad in ('"rssi":NaN', '"rssi":Infinity'):
        raw = f'{{"v":1,"tag":"A1B2C3D4E5F6","node":"N1",{bad},"ts":1,"seq":1}}'
        with pytest.raises(PayloadError):
            parse_json(raw)
