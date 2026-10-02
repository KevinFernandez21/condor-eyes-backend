"""Único puente del plano de video hacia el bus de agentes.

`MetadataPublisher` construye el `MetadataEnvelope` y valida el payload
completo con `ensure_metadata_only` ANTES de publicar, de modo que un frame,
array o buffer codificado nunca llega al `MetadataHub`. Los hilos del
pipeline son síncronos; el hub es asíncrono, por lo que la publicación se
programa en el event loop de los agentes con un tope de mensajes en vuelo.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from typing import Any

from bus import MetadataEnvelope, MetadataHub, Topic

from .health import StreamHealth
from .metadata import FrameMetadata, ensure_metadata_only


class MetadataPublisher:
    """Publica metadata validada del pipeline en un `MetadataHub`."""

    def __init__(
        self,
        hub: MetadataHub,
        loop: asyncio.AbstractEventLoop,
        *,
        source: str = "pipeline",
        max_pending: int = 256,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending debe ser >= 1")
        self._hub = hub
        self._loop = loop
        self._source = source
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._pending = 0
        self._dropped = 0

    @property
    def pending(self) -> int:
        """Mensajes programados que el hub aún no terminó de publicar."""
        with self._lock:
            return self._pending

    @property
    def dropped(self) -> int:
        """Mensajes descartados por exceder `max_pending` (hub lento)."""
        with self._lock:
            return self._dropped

    def publish_metadata(self, meta: FrameMetadata) -> None:
        """Publica la metadata de un frame en `Topic.DETECTIONS`."""
        self.publish_payload(Topic.DETECTIONS, meta.stream_id, meta.to_payload())

    def publish_status(self, health: StreamHealth) -> None:
        """Publica el estado de un stream en `Topic.STREAM_STATUS`."""
        self.publish_payload(
            Topic.STREAM_STATUS, health.stream_id, health.to_payload(), droppable=False
        )

    def publish_payload(
        self,
        topic: Topic,
        stream_id: str | None,
        payload: Mapping[str, Any],
        *,
        droppable: bool = True,
    ) -> None:
        """Valida y publica; lanza TypeError si el payload contiene pixeles."""
        ensure_metadata_only(payload)
        envelope = MetadataEnvelope(
            source=self._source, stream_id=stream_id, payload=dict(payload)
        )
        with self._lock:
            if droppable and self._pending >= self._max_pending:
                self._dropped += 1
                return
            self._pending += 1
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._hub.publish(topic, envelope), self._loop
            )
        except RuntimeError:  # event loop cerrado
            self._release()
            raise
        future.add_done_callback(lambda _future: self._release())

    def _release(self) -> None:
        with self._lock:
            self._pending -= 1
