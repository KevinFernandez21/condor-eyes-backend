"""Adaptador en memoria del bus, determinista y sin dependencias pesadas.

Sirve como implementación de referencia de ``MetadataHub`` y como base para
pruebas. Solo transporta mensajes: la deduplicación, los reintentos y las
reglas de negocio pertenecen a la capa de agentes, no a este adaptador.

Política ante colas saturadas, por tópico:

- ``EVENTS`` y ``COMMANDS`` son **sin pérdida** (``lossless``): la cola supera
  su tamaño en vez de descartar, y se publica un evento en ``ERRORS``.
- El resto (detecciones, tracks, salud, estado de streams) son telemetría de
  "última lectura gana": se descarta el mensaje más antiguo y se publica un
  evento en ``ERRORS``.
- ``ERRORS`` descarta el más antiguo sin emitir otro evento (evita bucles).

Los eventos de saturación se limitan a uno en el primer descarte y luego uno
cada 100, con el total acumulado en el payload, para no inundar el bus.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Final

from .hub import (
    HubError,
    MetadataEnvelope,
    Topic,
    build_error_envelope,
    parse_topic,
    validate_envelope,
)

_CLOSED: Final = object()

LOSSLESS_TOPICS: Final = frozenset({Topic.EVENTS, Topic.COMMANDS})
"""Tópicos que nunca se descartan por saturación."""

OVERFLOW_REPORT_EVERY: Final = 100


class HubClosedError(HubError):
    """Se intentó usar un hub ya cerrado."""


class Subscription:
    """Iterador asíncrono de un consumidor sobre un tópico.

    Se registra en el hub al crearse (no en la primera iteración), por lo que
    ningún mensaje publicado tras ``subscribe`` se pierde.
    """

    def __init__(
        self,
        topic: Topic,
        queue_size: int,
        hub: InMemoryHub,
        *,
        lossless: bool = False,
    ) -> None:
        self.topic = topic
        self.lossless = lossless
        self._items: deque[MetadataEnvelope | object] = deque()
        self._size = queue_size
        self._wakeup = asyncio.Event()
        self._closed = False
        self._hub = hub

    @property
    def pending(self) -> int:
        """Mensajes encolados aún no entregados al consumidor."""
        return sum(1 for item in self._items if item is not _CLOSED)

    def _push(self, message: MetadataEnvelope) -> MetadataEnvelope | None:
        """Encola un mensaje.

        Devuelve el mensaje descartado (política de telemetría), ``message``
        si la cola sin pérdida superó su tamaño, o ``None`` si cupo.
        """
        if self._closed:
            return None
        overflow: MetadataEnvelope | None = None
        if len(self._items) >= self._size:
            if self.lossless:
                overflow = message
            else:
                dropped = self._items.popleft()
                assert isinstance(dropped, MetadataEnvelope)
                overflow = dropped
        self._items.append(message)
        self._wakeup.set()
        return overflow

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
    """Implementación de ``MetadataHub`` con colas en proceso.

    ``history`` conserva los últimos ``history_size`` mensajes publicados y
    está pensada para pruebas y depuración; ``0`` la desactiva.
    """

    def __init__(
        self,
        *,
        queue_size: int = 1024,
        history_size: int = 1000,
        lossless_topics: frozenset[Topic] = LOSSLESS_TOPICS,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size debe ser al menos 1")
        if history_size < 0:
            raise ValueError("history_size no puede ser negativo")
        self._queue_size = queue_size
        self._lossless_topics = lossless_topics
        self._subscriptions: defaultdict[Topic, list[Subscription]] = defaultdict(list)
        self._closed = False
        self.history: deque[tuple[Topic, MetadataEnvelope]] = deque(maxlen=history_size)
        self.stats: dict[str, int] = {"published": 0, "dropped": 0, "overflow": 0}

    def _ensure_open(self) -> None:
        if self._closed:
            raise HubClosedError("El hub está cerrado; no admite más operaciones")

    def subscribe(self, topic: Topic) -> Subscription:
        """Registra un consumidor del tópico."""
        self._ensure_open()
        parsed = parse_topic(topic)
        subscription = Subscription(
            parsed,
            self._queue_size,
            self,
            lossless=parsed in self._lossless_topics,
        )
        self._subscriptions[parsed].append(subscription)
        return subscription

    async def publish(self, topic: Topic, message: MetadataEnvelope) -> None:
        """Valida el mensaje y lo entrega a los suscriptores del tópico."""
        self._ensure_open()
        parsed = validate_envelope(topic, message)
        await self._dispatch(parsed, message)
        self.stats["published"] += 1

    async def _dispatch(self, topic: Topic, message: MetadataEnvelope) -> None:
        """Punto de extensión: entrega ya validada. Los adaptadores lo amplían."""
        if self.history.maxlen:
            self.history.append((topic, message))
        for subscription in list(self._subscriptions[topic]):
            affected = subscription._push(message)
            if affected is None:
                continue
            counter = "overflow" if subscription.lossless else "dropped"
            self.stats[counter] += 1
            await self._report_overflow(topic, affected, subscription.lossless)

    async def _report_overflow(
        self, topic: Topic, affected: MetadataEnvelope, lossless: bool
    ) -> None:
        if topic is Topic.ERRORS:
            return  # sin recursión: solo se cuenta
        total = self.stats["overflow" if lossless else "dropped"]
        if total != 1 and (total - 1) % OVERFLOW_REPORT_EVERY != 0:
            return
        policy = "lossless" if lossless else "drop_oldest"
        error = OverflowError(
            f"Cola saturada en '{topic.value}' (política {policy}); "
            f"{'se amplió la cola' if lossless else 'se descartó el mensaje más antiguo'}"
        )
        report = build_error_envelope(
            "hub", error, failed=affected, failed_topic=topic, stage="queue_overflow"
        )
        payload = {**report.payload, "policy": policy, "dropped_total": total}
        report = report.derive(
            "hub", payload, suffix=f"overflow/{total}", stream_id=affected.stream_id
        )
        await self._dispatch(Topic.ERRORS, report)

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
