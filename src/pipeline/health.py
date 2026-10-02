"""Estado de salud por stream y política de reconexión con backoff."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class StreamState(StrEnum):
    """Ciclo de vida observable de un stream."""

    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class StreamHealth:
    """Instantánea inmutable de la salud de un stream (solo metadata)."""

    stream_id: str
    state: StreamState
    last_frame_at: datetime | None = None
    retry_count: int = 0
    last_error: str | None = None
    total_reconnects: int = 0
    frames_received: int = 0
    frames_dropped: int = 0
    queue_depth: int = 0
    callback_errors: int = 0

    def to_payload(self) -> dict[str, Any]:
        """Serializa a tipos primitivos aptos para un MetadataEnvelope."""
        return {
            "stream_id": self.stream_id,
            "state": self.state.value,
            "last_frame_at": (
                self.last_frame_at.isoformat() if self.last_frame_at else None
            ),
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "total_reconnects": self.total_reconnects,
            "frames_received": self.frames_received,
            "frames_dropped": self.frames_dropped,
            "queue_depth": self.queue_depth,
            "callback_errors": self.callback_errors,
        }


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Backoff exponencial acotado con jitter opcional."""

    initial: float = 1.0
    factor: float = 2.0
    maximum: float = 30.0
    jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.initial <= 0:
            raise ValueError("initial debe ser mayor que 0")
        if self.factor < 1:
            raise ValueError("factor debe ser >= 1")
        if self.maximum < self.initial:
            raise ValueError("maximum no puede ser menor que initial")
        if not 0 <= self.jitter < 1:
            raise ValueError("jitter debe estar en [0, 1)")

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Espera antes del intento `attempt` (0 = primer reintento)."""
        base = min(self.initial * self.factor ** max(attempt, 0), self.maximum)
        if self.jitter == 0:
            return base
        generator = rng or random
        return base * (1 + generator.uniform(-self.jitter, self.jitter))
