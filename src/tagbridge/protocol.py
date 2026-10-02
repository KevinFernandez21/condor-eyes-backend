"""Formato del anuncio BLE del tag XIAO ESP32-C6 (datos de fabricante).

Debe coincidir con `firmware/xiao_c6_tag/xiao_c6_tag.ino`. Tras el company ID
(0xFFFF, reservado para pruebas por el Bluetooth SIG) el anuncio lleva 14 bytes
little-endian:

    magic "CE" (2) | versión (1) | tag (6) | seq u32 (4) | batería u8 (1)

La batería `0xFF` significa desconocida.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

COMPANY_ID = 0xFFFF
MAGIC = b"CE"
VERSION = 1
BATTERY_UNKNOWN = 0xFF
SEQ_MASK = 0xFFFFFFFF
_STRUCT = struct.Struct("<2sB6sIB")
_TAG_RE = re.compile(r"^[0-9A-Fa-f]{12}$")


@dataclass(frozen=True)
class TagPayload:
    tag: str
    seq: int
    battery_pct: int | None


def normalize_tag(tag: str) -> str:
    """ID de tag canónico: 12 dígitos hex en mayúsculas."""
    if not _TAG_RE.match(tag):
        raise ValueError(f"ID de tag inválido (se esperan 12 dígitos hex): {tag!r}")
    return tag.upper()


def encode_tag_payload(tag: str, seq: int, battery_pct: int | None) -> bytes:
    if not 0 <= seq <= SEQ_MASK:
        raise ValueError(f"seq fuera de rango uint32: {seq}")
    if battery_pct is not None and not 0 <= battery_pct <= 100:
        raise ValueError(f"batería fuera de 0..100: {battery_pct}")
    bat = BATTERY_UNKNOWN if battery_pct is None else battery_pct
    return _STRUCT.pack(MAGIC, VERSION, bytes.fromhex(normalize_tag(tag)), seq, bat)


def decode_tag_payload(raw: bytes) -> TagPayload | None:
    """Decodifica los datos de fabricante; `None` si no son de un tag Condor."""
    if len(raw) != _STRUCT.size:
        return None
    magic, version, tag, seq, bat = _STRUCT.unpack(raw)
    if magic != MAGIC or version != VERSION:
        return None
    battery = None if bat == BATTERY_UNKNOWN or bat > 100 else bat
    return TagPayload(tag=tag.hex().upper(), seq=seq, battery_pct=battery)
