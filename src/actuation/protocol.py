"""Protocolo de comandos y acks entre el host (Jetson) y el nodo ESP32-S3 Pan-Tilt.

Es el protocolo de hardware: el firmware del ESP32-S3 debe implementar
exactamente estas tramas (ver ``docs/pan-tilt.md``).

Formato de trama (una por línea; el transporte añade el ``\\n`` final)::

    <JSON compacto con claves ordenadas>*<CRC16 en 4 hex mayúsculas>

El CRC es CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) calculado sobre los
bytes exactos del JSON, es decir, todo lo que precede al último ``*``.

Los números de secuencia son uint16 con aritmética modular (RFC 1982): un
comando solo es válido si su ``seq`` es estrictamente más nuevo que el último
aceptado. Los comandos de parada de emergencia son la excepción: se atienden
siempre.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

SEQ_MODULUS = 1 << 16
_SEQ_HALF = 1 << 15


class ProtocolError(ValueError):
    """Trama corrupta, mal formada o con campos inválidos."""


class AckStatus(StrEnum):
    """Resultado que el nodo reporta para un comando."""

    OK = "ok"  # aceptado y aplicado
    DUPLICATE = "dup"  # mismo seq que el último aceptado: reconocido sin mover
    STALE = "stale"  # seq anterior al último aceptado: ignorado sin mover
    ESTOP = "estop"  # rechazado porque la parada de emergencia está activa
    REJECTED = "rejected"  # campos fuera de rango o no finitos


class NodeState(StrEnum):
    """Estado de la máquina de estados del nodo."""

    IDLE = "idle"
    MOVING = "moving"
    FAILSAFE = "failsafe"  # watchdog de comunicación vencido: congelado en sitio
    ESTOP = "estop"  # parada de emergencia enclavada


def seq_is_newer(candidate: int, reference: int) -> bool:
    """True si ``candidate`` es estrictamente posterior a ``reference`` (mod 2^16)."""
    diff = (candidate - reference) % SEQ_MODULUS
    return 0 < diff < _SEQ_HALF


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, sin reflejo ni xorout)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
            )
    return crc


def _check_seq(seq: int) -> None:
    if not 0 <= seq < SEQ_MODULUS:
        raise ValueError(f"seq debe estar en [0, {SEQ_MODULUS - 1}], recibido {seq}")


@dataclass(frozen=True, slots=True)
class Move:
    """Ir al ángulo objetivo (grados respecto al neutro) sin superar ``speed_dps``."""

    seq: int
    pan_deg: float
    tilt_deg: float
    speed_dps: float

    def __post_init__(self) -> None:
        _check_seq(self.seq)
        for name in ("pan_deg", "tilt_deg", "speed_dps"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} debe ser finito")
        if self.speed_dps <= 0:
            raise ValueError("speed_dps debe ser positivo")


@dataclass(frozen=True, slots=True)
class EmergencyStop:
    """Congela el movimiento en la posición actual y enclava el estado ESTOP."""

    seq: int

    def __post_init__(self) -> None:
        _check_seq(self.seq)


@dataclass(frozen=True, slots=True)
class ClearEstop:
    """Libera el enclavamiento de ESTOP. No mueve los servos."""

    seq: int

    def __post_init__(self) -> None:
        _check_seq(self.seq)


@dataclass(frozen=True, slots=True)
class Heartbeat:
    """Mantiene vivo el watchdog del nodo y pide telemetría sin mover."""

    seq: int

    def __post_init__(self) -> None:
        _check_seq(self.seq)


Command = Move | EmergencyStop | ClearEstop | Heartbeat


@dataclass(frozen=True, slots=True)
class Ack:
    """Respuesta del nodo a un comando, con telemetría de posición medida."""

    seq: int
    status: AckStatus
    state: NodeState
    pan_deg: float
    tilt_deg: float
    node_ms: int

    def __post_init__(self) -> None:
        _check_seq(self.seq)


def _r(value: float) -> float:
    return round(float(value), 2)


def _seal(body: dict[str, Any]) -> bytes:
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    return raw + b"*" + f"{crc16_ccitt(raw):04X}".encode("ascii")


def _unseal(frame: bytes) -> dict[str, Any]:
    body, sep, tail = frame.rpartition(b"*")
    if not sep:
        raise ProtocolError("trama sin separador de CRC")
    try:
        received = int(tail.decode("ascii"), 16)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError("CRC ilegible") from exc
    if len(tail) != 4 or received != crc16_ccitt(body):
        raise ProtocolError("CRC inválido")
    try:
        data = json.loads(body.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError("cuerpo JSON inválido") from exc
    if not isinstance(data, dict):
        raise ProtocolError("el cuerpo debe ser un objeto JSON")
    return data


def _num(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProtocolError(f"campo '{key}' ausente o no numérico")
    if not math.isfinite(value):
        raise ProtocolError(f"campo '{key}' no finito")
    return float(value)


def _int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"campo '{key}' ausente o no entero")
    return value


def encode_command(cmd: Command) -> bytes:
    """Serializa un comando a trama (sin el ``\\n`` final)."""
    if isinstance(cmd, Move):
        body: dict[str, Any] = {
            "cmd": "move",
            "seq": cmd.seq,
            "pan": _r(cmd.pan_deg),
            "tilt": _r(cmd.tilt_deg),
            "speed": _r(cmd.speed_dps),
        }
    elif isinstance(cmd, EmergencyStop):
        body = {"cmd": "estop", "seq": cmd.seq}
    elif isinstance(cmd, ClearEstop):
        body = {"cmd": "clear", "seq": cmd.seq}
    else:
        body = {"cmd": "hb", "seq": cmd.seq}
    return _seal(body)


def decode_command(frame: bytes) -> Command:
    """Parsea y valida una trama de comando. Solo lanza :class:`ProtocolError`."""
    data = _unseal(frame)
    kind = data.get("cmd")
    try:
        seq = _int(data, "seq")
        if kind == "move":
            return Move(
                seq=seq,
                pan_deg=_num(data, "pan"),
                tilt_deg=_num(data, "tilt"),
                speed_dps=_num(data, "speed"),
            )
        if kind == "estop":
            return EmergencyStop(seq=seq)
        if kind == "clear":
            return ClearEstop(seq=seq)
        if kind == "hb":
            return Heartbeat(seq=seq)
    except ValueError as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(str(exc)) from exc
    raise ProtocolError(f"comando desconocido: {kind!r}")


def encode_ack(ack: Ack) -> bytes:
    """Serializa un ack a trama (sin el ``\\n`` final)."""
    return _seal(
        {
            "ack": ack.seq,
            "st": ack.status.value,
            "state": ack.state.value,
            "pan": _r(ack.pan_deg),
            "tilt": _r(ack.tilt_deg),
            "ms": ack.node_ms,
        }
    )


def decode_ack(frame: bytes) -> Ack:
    """Parsea y valida una trama de ack. Solo lanza :class:`ProtocolError`."""
    data = _unseal(frame)
    try:
        return Ack(
            seq=_int(data, "ack"),
            status=AckStatus(str(data.get("st"))),
            state=NodeState(str(data.get("state"))),
            pan_deg=_num(data, "pan"),
            tilt_deg=_num(data, "tilt"),
            node_ms=_int(data, "ms"),
        )
    except ProtocolError:
        raise
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
