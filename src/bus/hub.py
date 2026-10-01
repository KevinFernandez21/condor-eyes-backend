"""Contrato tipado que aísla AgentScope del dominio de Condor Eye.

Este módulo transporta exclusivamente metadata serializable. Los frames y
buffers NVMM pertenecen al pipeline compartido y no forman parte del bus.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol


class Topic(StrEnum):
    """Canales permitidos para la coordinación entre agentes."""

    STREAM_STATUS = "stream.status"
    DETECTIONS = "vision.detections"
    TRACKS = "vision.tracks"
    EVENTS = "events"
    HEALTH = "system.health"
    COMMANDS = "system.commands"


@dataclass(frozen=True, slots=True)
class MetadataEnvelope:
    """Mensaje serializable que nunca contiene frames de video."""

    source: str
    payload: Mapping[str, Any]
    stream_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MetadataHub(Protocol):
    """Interfaz que deberá adaptar el mecanismo de mensajería de AgentScope."""

    async def publish(self, topic: Topic, message: MetadataEnvelope) -> None:
        """Publica metadata en un canal tipado."""

    def subscribe(self, topic: Topic) -> AsyncIterator[MetadataEnvelope]:
        """Suscribe un consumidor a un canal tipado."""
