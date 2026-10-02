"""Hub con contadores por tópico y sumideros en memoria para el sistema local.

``InstrumentedHub`` es un ``InMemoryHub`` (mismo contrato) que además cuenta
mensajes por tópico; así ``SystemApp.snapshot()`` y la API de observabilidad
leen estadísticas sin recorrer el historial.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import Any

from bus import InMemoryHub, MetadataEnvelope, Topic
from bus.memory import LOSSLESS_TOPICS


class InstrumentedHub(InMemoryHub):
    """``InMemoryHub`` que cuenta cuántos mensajes pasan por cada tópico."""

    def __init__(
        self,
        *,
        queue_size: int = 1024,
        history_size: int = 1000,
        lossless_topics: frozenset[Topic] = LOSSLESS_TOPICS,
    ) -> None:
        super().__init__(
            queue_size=queue_size,
            history_size=history_size,
            lossless_topics=lossless_topics,
        )
        self._counts: dict[Topic, int] = dict.fromkeys(Topic, 0)
        self._last_at: dict[Topic, datetime | None] = dict.fromkeys(Topic)

    async def _dispatch(self, topic: Topic, message: MetadataEnvelope) -> None:
        self._counts[topic] += 1
        self._last_at[topic] = datetime.now(UTC)
        await super()._dispatch(topic, message)

    def topic_stats(self) -> dict[str, dict[str, Any]]:
        """Mensajes entregados y marca del último, por cada ``Topic``."""
        return {
            topic.value: {
                "published": self._counts[topic],
                "last_at": last.isoformat() if (last := self._last_at[topic]) else None,
            }
            for topic in Topic
        }


class _RecentBuffer:
    def __init__(self, limit: int) -> None:
        self._items: deque[MetadataEnvelope] = deque(maxlen=limit)
        self.total = 0

    def add(self, envelope: MetadataEnvelope) -> None:
        self.total += 1
        self._items.append(envelope)

    @property
    def recent(self) -> list[MetadataEnvelope]:
        return list(self._items)


class MemoryEventSink(_RecentBuffer):
    """Puerto ``EventSink`` en memoria (en producción: clips + SQLite WAL)."""

    async def save(self, envelope: MetadataEnvelope) -> None:
        self.add(envelope)


class MemoryAlertSink(_RecentBuffer):
    """Puerto ``AlertSink`` en memoria (en producción: API HTTP/WS)."""

    async def send(self, envelope: MetadataEnvelope) -> None:
        self.add(envelope)
