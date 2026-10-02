"""Sumideros en memoria (``EventSink``/``AlertSink``) del sistema local.

Las estadísticas por tópico las lleva únicamente el ``BusTap`` de ``comms``.
"""

from __future__ import annotations

from collections import deque

from bus import MetadataEnvelope


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
