"""Adaptador en memoria del bus, determinista y sin dependencias pesadas.

Sirve como implementación de referencia de ``MetadataHub`` y como base para
pruebas. Solo transporta mensajes: la deduplicación, los reintentos y las
reglas de negocio pertenecen a la capa de agentes, no a este adaptador.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Final

from .hub import HubError, MetadataEnvelope, Topic, parse_topic, validate_envelope

_CLOSED: Final = object()


class HubClosedError(HubError):
    """Se intentó usar un hub ya cerrado."""


class Subscription:
    """Iterador asíncrono de un consumidor sobre un tópico.

    Se registra en el hub al crearse (no en la primera iteración), por lo que
    ningún mensaje publicado tras ``subscribe`` se pierde.
    """

    def __init__(self, topic: Topic, queue_size: int, hub: InMemoryHub) -> None:
        self.topic = topic
        self._items: deque[MetadataEnvelope | object] = deque()
        self._size = queue_size
        self._wakeup = asyncio.Event()
        self._closed = False
        self._hub = hub

    def _push(self, message: MetadataEnvelope) -> bool:
        """Encola un mensaje; devuelve ``True`` si tuvo que descartar uno."""
        if self._closed:
            return False
        dropped = False
        if len(self._items) >= self._size:
            self._items.popleft()
            dropped = True
        self._items.append(message)
        self._wakeup.set()
        return dropped

    def close(self) -> None:
        """Termina la iteración tras entregar lo ya encolado."""
        if self._closed:
            return
        self._closed = True
        self._items.append(_CLOSED)
        self._wakeup.set()
        self._hub._detach(self)

    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> MetadataEnvelope:
        while True:
            if self._items:
                item = self._items.popleft()
                if item is _CLOSED:
                    self._items.append(_CLOSED)
                    raise StopAsyncIteration
                assert isinstance(item, MetadataEnvelope)
                return item
            self._wakeup.clear()
            await self._wakeup.wait()


class InMemoryHub:
    """Implementación de ``MetadataHub`` con colas en proceso."""

    def __init__(self, *, queue_size: int = 1024) -> None:
        if queue_size < 1:
            raise ValueError("queue_size debe ser al menos 1")
        self._queue_size = queue_size
        self._subscriptions: defaultdict[Topic, list[Subscription]] = defaultdict(
            list
        )
        self._closed = False
        self.history: list[tuple[Topic, MetadataEnvelope]] = []
        self.stats: dict[str, int] = {"published": 0, "dropped": 0}

    def _ensure_open(self) -> None:
        if self._closed:
            raise HubClosedError("El hub está cerrado; no admite más operaciones")

    def subscribe(self, topic: Topic) -> Subscription:
        """Registra un consumidor del tópico."""
        self._ensure_open()
        subscription = Subscription(parse_topic(topic), self._queue_size, self)
        self._subscriptions[subscription.topic].append(subscription)
        return subscription

    async def publish(self, topic: Topic, message: MetadataEnvelope) -> None:
        """Valida el mensaje y lo entrega a los suscriptores del tópico."""
        self._ensure_open()
        parsed = validate_envelope(topic, message)
        await self._dispatch(parsed, message)
        self.stats["published"] += 1

    async def _dispatch(self, topic: Topic, message: MetadataEnvelope) -> None:
        """Punto de extensión: entrega ya validada. Los adaptadores lo amplían."""
        self.history.append((topic, message))
        for subscription in list(self._subscriptions[topic]):
            if subscription._push(message):
                self.stats["dropped"] += 1

    def _detach(self, subscription: Subscription) -> None:
        peers = self._subscriptions[subscription.topic]
        if subscription in peers:
            peers.remove(subscription)

    async def close(self) -> None:
        """Cierra el hub; los consumidores terminan tras vaciar sus colas."""
        if self._closed:
            return
        self._closed = True
        for subscriptions in list(self._subscriptions.values()):
            for subscription in list(subscriptions):
                subscription.close()
