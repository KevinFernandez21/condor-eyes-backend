"""Tap del bus: observador de solo lectura de toda la metadata publicada.

Se suscribe a **todos** los tópicos mediante la interfaz tipada ``MetadataHub``
(nunca AgentScope directo) y conserva únicamente metadata ya validada por el
bus: buffers acotados (eventos, decisiones, últimos N por tópico) y
estadísticas por tópico. No guarda frames, imágenes ni embeddings porque el bus
no los transporta.

El tap es un consumidor más: su procesamiento es síncrono y barato, y los
oyentes (clientes WebSocket) se invocan sin esperar, así que ni un cliente
lento ni uno muerto pueden retrasar a los agentes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from bus import (
    MetadataEnvelope,
    MetadataHub,
    MetadataSubscription,
    Topic,
    envelope_to_dict,
)

logger = logging.getLogger(__name__)

Listener = Callable[[Topic, dict[str, Any]], None]
"""Oyente síncrono y no bloqueante: recibe cada mensaje ya serializado."""

_MAX_WINDOW_SAMPLES = 100_000


def payload_zone(data: dict[str, Any]) -> str | None:
    """Zona declarada en el payload (``zone_id`` o ``zone``), si existe."""
    payload = data.get("payload") or {}
    zone = payload.get("zone_id", payload.get("zone"))
    return None if zone is None else str(zone)


def payload_stream(data: dict[str, Any]) -> str | None:
    """Cámara del mensaje: ``stream_id`` del envelope o del payload."""
    stream = data.get("stream_id")
    if stream is None:
        stream = (data.get("payload") or {}).get("stream_id")
    return None if stream is None else str(stream)


def _matches(data: dict[str, Any], stream_id: str | None, zone: str | None) -> bool:
    if stream_id is not None and payload_stream(data) != stream_id:
        return False
    return zone is None or payload_zone(data) == zone


def is_decision(data: dict[str, Any]) -> bool:
    """Una decisión de fusión viaja por ``events`` con ``decision_id``."""
    return "decision_id" in (data.get("payload") or {})


class _TopicStats:
    def __init__(self) -> None:
        self.count = 0
        self.drops = 0
        self.last_message_at: str | None = None
        self.window: deque[float] = deque(maxlen=_MAX_WINDOW_SAMPLES)


class BusTap:
    """Anillos acotados y estadísticas por tópico sobre un ``MetadataHub``."""

    def __init__(
        self,
        hub: MetadataHub,
        *,
        events_size: int = 500,
        decisions_size: int = 500,
        topic_size: int = 50,
        window_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        for name, value in (
            ("events_size", events_size),
            ("decisions_size", decisions_size),
            ("topic_size", topic_size),
        ):
            if value < 1:
                raise ValueError(f"{name} debe ser al menos 1")
        if window_s <= 0:
            raise ValueError("window_s debe ser positivo")
        self._hub = hub
        self._window_s = window_s
        self._clock = clock
        self._wall = wall
        self._events: deque[dict[str, Any]] = deque(maxlen=events_size)
        self._decisions: deque[dict[str, Any]] = deque(maxlen=decisions_size)
        self._recent: dict[Topic, deque[dict[str, Any]]] = {
            topic: deque(maxlen=topic_size) for topic in Topic
        }
        self._stats: dict[Topic, _TopicStats] = {topic: _TopicStats() for topic in Topic}
        self._heartbeats: dict[str, str] = {}
        self._restarts: dict[str, int] = {}
        self._listeners: list[Listener] = []
        self._subscriptions: list[MetadataSubscription] = []
        self._tasks: list[asyncio.Task[None]] = []

    # -- ciclo de vida --

    async def start(self) -> None:
        """Suscribe el tap a todos los tópicos (antes que los productores)."""
        if self._tasks:
            raise RuntimeError("El tap ya está en ejecución")
        for topic in Topic:
            subscription = self._hub.subscribe(topic)
            self._subscriptions.append(subscription)
            self._tasks.append(
                asyncio.create_task(self._consume(topic, subscription), name=f"tap:{topic}")
            )

    async def stop(self) -> None:
        for subscription in self._subscriptions:
            subscription.close()
        if self._tasks:
            _, pending = await asyncio.wait(self._tasks, timeout=2.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        self._subscriptions.clear()

    async def _consume(self, topic: Topic, subscription: MetadataSubscription) -> None:
        async for envelope in subscription:
            try:
                self.record(topic, envelope)
            except Exception:
                logger.exception("El tap descartó un mensaje de '%s'", topic.value)

    # -- registro --

    def record(self, topic: Topic, envelope: MetadataEnvelope) -> dict[str, Any]:
        """Valida, serializa y registra un mensaje; avisa a los oyentes."""
        data = envelope_to_dict(topic, envelope)  # revalida: finito, sin binarios
        topic = Topic(topic)
        stats = self._stats[topic]
        stats.count += 1
        stats.last_message_at = self._wall().isoformat()
        stats.window.append(self._clock())
        self._recent[topic].append(data)
        payload = data["payload"]
        if topic is Topic.EVENTS:
            self._events.append(data)
            if is_decision(data):
                self._decisions.append(data)
        elif topic is Topic.HEALTH:
            self._heartbeats[envelope.source] = data["created_at"]
        elif topic is Topic.COMMANDS and payload.get("action") == "restart":
            target = str(payload.get("target"))
            self._restarts[target] = self._restarts.get(target, 0) + 1
        elif (
            topic is Topic.ERRORS
            and payload.get("stage") == "queue_overflow"
            and payload.get("policy") == "drop_oldest"
        ):
            failed = payload.get("failed_topic")
            if failed in {t.value for t in Topic}:
                self._stats[Topic(failed)].drops += 1
        for listener in tuple(self._listeners):
            try:
                listener(topic, data)
            except Exception:
                logger.exception("Oyente del tap falló; se ignora")
        return data

    # -- oyentes --

    def add_listener(self, listener: Listener) -> Callable[[], None]:
        """Registra un oyente y devuelve la función que lo quita."""
        self._listeners.append(listener)

        def remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return remove

    # -- lectura (siempre el más nuevo primero) --

    @staticmethod
    def _latest(
        ring: deque[dict[str, Any]], limit: int, stream_id: str | None, zone: str | None
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        # instantánea: tuple() copia en C sin ceder el GIL; otro hilo puede estar añadiendo
        for data in reversed(tuple(ring)):
            if _matches(data, stream_id, zone):
                out.append(data)
                if len(out) >= limit:
                    break
        return out

    def recent(self, topic: Topic, limit: int = 50) -> list[dict[str, Any]]:
        return self._latest(self._recent[Topic(topic)], limit, None, None)

    def events(
        self, limit: int = 50, *, stream_id: str | None = None, zone: str | None = None
    ) -> list[dict[str, Any]]:
        return self._latest(self._events, limit, stream_id, zone)

    def decisions(
        self, limit: int = 50, *, stream_id: str | None = None, zone: str | None = None
    ) -> list[dict[str, Any]]:
        return self._latest(self._decisions, limit, stream_id, zone)

    def topic_stats(self) -> list[dict[str, Any]]:
        """Conteo, tasa en ventana deslizante, descartes y último mensaje."""
        now = self._clock()
        out: list[dict[str, Any]] = []
        for topic in Topic:
            stats = self._stats[topic]
            window = stats.window
            try:
                while window and window[0] <= now - self._window_s:
                    window.popleft()
            except IndexError:  # otro lector la vació entre la comprobación y el pop
                pass
            out.append(
                {
                    "topic": topic.value,
                    "count": stats.count,
                    "rate_per_s": round(len(window) / self._window_s, 3),
                    "window_s": self._window_s,
                    "drops": stats.drops,
                    "last_message_at": stats.last_message_at,
                }
            )
        return out

    def heartbeats(self) -> dict[str, str]:
        """Último ``system.health`` visto por fuente (marca de tiempo del envelope)."""
        return dict(self._heartbeats)

    def restarts(self) -> dict[str, int]:
        """Comandos de reinicio del supervisor vistos por rol objetivo."""
        return dict(self._restarts)

    def latest_health(self) -> dict[str, dict[str, Any]]:
        """Último payload de salud por fuente, para sistemas sin runtime."""
        out: dict[str, dict[str, Any]] = {}
        for data in self._recent[Topic.HEALTH]:
            out[data["source"]] = data["payload"]
        return out
