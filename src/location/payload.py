"""Contrato de mensajes entre nodos de zona ESP32-S3 y la localización.

Hay dos codificaciones equivalentes de la misma observación (versión 1):

* JSON (UTF-8): legible, para BLE/Wi-Fi/MQTT.
* Binario compacto de 26 bytes little-endian con CRC16: para LoRa/LoRaWAN.

Ver `docs/location.md` para el esquema completo.
"""

from __future__ import annotations

import json
import re
import struct
from datetime import UTC, datetime
from typing import Any

from .models import Channel, TagObservation

PAYLOAD_VERSION = 1

# ver u8, flags u8, tag 6B, nodo u16, ts_ms u64, seq u32, rssi i8, bat u8, crc u16
_BODY = struct.Struct("<BB6sHQIbB")
_CRC = struct.Struct("<H")
BINARY_SIZE = _BODY.size + _CRC.size
# Un payload legítimo mide ~120 B; el tope evita trabajo y recursión con entradas hostiles.
MAX_JSON_BYTES = 1024

_TAG_RE = re.compile(r"[0-9A-Fa-f]{12}")
_NODE_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")
_BINARY_NODE_RE = re.compile(r"N(\d{1,5})")
_BATTERY_UNKNOWN = 0xFF
_MAX_SEQ = 2**32 - 1
_CHANNEL_BITS = {Channel.BLE: 0, Channel.LORA: 1}


class PayloadError(ValueError):
    """El payload recibido no cumple el contrato (malformado)."""


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
            )
    return crc


def _require_int(raw: dict[str, Any], key: str) -> int:
    if key not in raw:
        raise PayloadError(f"falta el campo obligatorio '{key}'")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PayloadError(f"el campo '{key}' debe ser un entero")
    return value


def _timestamp_from_ms(ts_ms: int) -> datetime:
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise PayloadError("el campo 'ts' está fuera de rango") from exc


def parse_json(raw: str | bytes | bytearray) -> TagObservation:
    """Valida y convierte un payload JSON v1 en una observación."""
    if not isinstance(raw, str | bytes | bytearray):
        raise PayloadError(f"tipo de entrada inválido: {type(raw).__name__}")
    if len(raw) > MAX_JSON_BYTES:
        raise PayloadError(
            f"el payload JSON excede el máximo de {MAX_JSON_BYTES} bytes"
        )
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError, TypeError) as exc:
        # UnicodeDecodeError es subclase de ValueError.
        raise PayloadError("el payload no es JSON válido") from exc
    if not isinstance(data, dict):
        raise PayloadError("el payload JSON debe ser un objeto")

    version = _require_int(data, "v")
    if version != PAYLOAD_VERSION:
        raise PayloadError(f"versión de payload no soportada: {version}")

    tag = data.get("tag")
    if "tag" not in data:
        raise PayloadError("falta el campo obligatorio 'tag'")
    if not isinstance(tag, str) or not _TAG_RE.fullmatch(tag):
        raise PayloadError("el campo 'tag' debe ser hexadecimal de 12 dígitos")

    node = data.get("node")
    if "node" not in data:
        raise PayloadError("falta el campo obligatorio 'node'")
    if not isinstance(node, str) or not _NODE_RE.fullmatch(node):
        raise PayloadError("el campo 'node' tiene un formato inválido")

    rssi = _require_int(data, "rssi")
    ts_ms = _require_int(data, "ts")
    seq = _require_int(data, "seq")
    if not 0 <= seq <= _MAX_SEQ:
        raise PayloadError("el campo 'seq' debe caber en 32 bits sin signo")

    battery: int | None = None
    if data.get("bat") is not None:
        battery = _require_int(data, "bat")

    try:
        channel = Channel(data.get("ch", Channel.BLE.value))
    except ValueError as exc:
        raise PayloadError(f"canal desconocido: {data.get('ch')!r}") from exc

    return TagObservation(
        tag_id=tag.upper(),
        node_id=node,
        rssi_dbm=rssi,
        timestamp=_timestamp_from_ms(ts_ms),
        sequence=seq,
        battery_pct=battery,
        channel=channel,
    )


def encode_json(obs: TagObservation) -> str:
    """Serializa una observación al JSON v1 (útil para simuladores y pruebas)."""
    data: dict[str, Any] = {
        "v": PAYLOAD_VERSION,
        "ch": obs.channel.value,
        "tag": obs.tag_id,
        "node": obs.node_id,
        "rssi": obs.rssi_dbm,
        "ts": round(obs.timestamp.timestamp() * 1000),
        "seq": obs.sequence,
    }
    if obs.battery_pct is not None:
        data["bat"] = obs.battery_pct
    return json.dumps(data, separators=(",", ":"))


def encode_binary(obs: TagObservation) -> bytes:
    """Serializa al formato binario compacto de 26 bytes."""
    match = _BINARY_NODE_RE.fullmatch(obs.node_id)
    if match is None or int(match.group(1)) > 0xFFFF:
        raise ValueError(
            f"el id de nodo '{obs.node_id}' no es codificable en binario "
            "(se espera 'N' seguido de un número de 0 a 65535)"
        )
    battery = _BATTERY_UNKNOWN if obs.battery_pct is None else obs.battery_pct
    body = _BODY.pack(
        PAYLOAD_VERSION,
        _CHANNEL_BITS[obs.channel],
        bytes.fromhex(obs.tag_id),
        int(match.group(1)),
        round(obs.timestamp.timestamp() * 1000),
        obs.sequence,
        obs.rssi_dbm,
        battery,
    )
    return body + _CRC.pack(crc16_ccitt(body))


def parse_binary(raw: bytes) -> TagObservation:
    """Valida tipo, longitud, CRC y versión; devuelve la observación."""
    if not isinstance(raw, bytes | bytearray):
        raise PayloadError(f"tipo de entrada inválido: {type(raw).__name__}")
    raw = bytes(raw)
    if len(raw) != BINARY_SIZE:
        raise PayloadError(
            f"longitud inválida: se esperaban {BINARY_SIZE} bytes y llegaron {len(raw)}"
        )
    body, (crc,) = raw[: _BODY.size], _CRC.unpack(raw[_BODY.size :])
    if crc16_ccitt(body) != crc:
        raise PayloadError("CRC inválido: el paquete llegó corrupto")
    version, flags, tag, node_num, ts_ms, seq, rssi, battery = _BODY.unpack(body)
    if version != PAYLOAD_VERSION:
        raise PayloadError(f"versión de payload no soportada: {version}")
    channel_bits = flags & 0b11
    channel = {v: k for k, v in _CHANNEL_BITS.items()}.get(channel_bits)
    if channel is None:
        raise PayloadError(f"canal desconocido en flags: {channel_bits}")
    return TagObservation(
        tag_id=tag.hex().upper(),
        node_id=f"N{node_num:04d}",
        rssi_dbm=rssi,
        timestamp=_timestamp_from_ms(ts_ms),
        sequence=seq,
        battery_pct=None if battery == _BATTERY_UNKNOWN else battery,
        channel=channel,
    )
