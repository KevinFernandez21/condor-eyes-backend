"""Transportes entre el host y el nodo Pan-Tilt.

El adaptador solo conoce la interfaz :class:`Transport`: un transporte entrega y
recibe tramas completas (sin el ``\\n`` de línea). El transporte serie importa
``pyserial`` de forma diferida para que el módulo cargue sin hardware.
"""

from __future__ import annotations

from typing import Any, Protocol


class TransportError(RuntimeError):
    """Fallo del transporte (puerto no disponible, dependencia ausente)."""


class Transport(Protocol):
    """Canal de tramas hacia el nodo. ``kind`` distingue simulado de hardware real."""

    kind: str  # "simulated" | "serial"

    def send(self, frame: bytes) -> None:
        """Envía una trama completa; no bloquea."""

    def recv(self) -> list[bytes]:
        """Devuelve las tramas completas recibidas hasta ahora; no bloquea."""

    def close(self) -> None:
        """Libera el canal."""


class SerialTransport:
    """Transporte por puerto serie (USB CDC del ESP32-S3), tramas terminadas en ``\\n``."""

    kind = "serial"

    def __init__(
        self, port: str, baudrate: int = 115200, max_frame_bytes: int = 256
    ) -> None:
        try:
            import serial  # type: ignore[import-untyped,import-not-found,unused-ignore]
        except ImportError as exc:
            raise TransportError(
                "SerialTransport requiere pyserial; instálalo con `uv add pyserial`"
            ) from exc
        self._max = max_frame_bytes
        self._buffer = bytearray()
        self._discarding = False
        # timeout=0: lecturas no bloqueantes; write_timeout evita colgar el lazo.
        self._port: Any = serial.Serial(port, baudrate, timeout=0, write_timeout=0.1)

    def send(self, frame: bytes) -> None:
        try:
            self._port.write(frame + b"\n")
        except Exception as exc:
            raise TransportError(
                f"fallo al escribir en el puerto serie: {exc}"
            ) from exc

    def recv(self) -> list[bytes]:
        try:
            waiting = self._port.in_waiting
            if waiting:
                self._buffer += self._port.read(waiting)
        except Exception as exc:
            raise TransportError(f"fallo al leer del puerto serie: {exc}") from exc
        frames: list[bytes] = []
        while (idx := self._buffer.find(b"\n")) >= 0:
            line = bytes(self._buffer[:idx]).strip()
            del self._buffer[: idx + 1]
            if self._discarding:  # cola de una línea que ya desbordó el límite
                self._discarding = False
                continue
            if len(line) > self._max:  # línea demasiado larga: basura, se descarta
                continue
            if line:
                frames.append(line)
        if len(self._buffer) > self._max:
            # Basura sin salto de línea: descartar y seguir ignorando hasta el
            # próximo salto de línea para que su cola no parezca una trama nueva.
            self._buffer.clear()
            self._discarding = True
        return frames

    def close(self) -> None:
        self._port.close()
