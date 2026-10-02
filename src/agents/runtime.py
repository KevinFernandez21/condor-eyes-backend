"""Runtime que ejecuta la ruta multiagente sobre el bus de metadata.

Responsabilidades (ciclo de vida, no negocio):

- instanciar un ``RoleWorker`` por rol de ``MULTIAGENT_ROUTE`` (y por cámara en
  los roles no singleton);
- arrancar y detener de forma ordenada, con vaciado de colas y rollback si un
  rol falla al iniciar;
- reintentar con backoff, publicar eventos de error y deduplicar por
  ``event_id`` para que reentregas no repitan efectos secundarios;
- exponer salud (solo metadata) y un puente seguro entre hilos para que el
  pipeline de video publique metadata sin acceder al bus asyncio.

Las reglas de negocio viven en los manejadores (``agents.handlers``); el
transporte vive en ``bus``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from bus import (
    MetadataEnvelope,
    MetadataHub,
    Topic,
    build_error_envelope,
    parse_topic,
)

from .base import AgentRoute
from .route import MULTIAGENT_ROUTE, validate_route

logger = logging.getLogger(__name__)

Outgoing = tuple[Topic, MetadataEnvelope]
Sleep = Callable[[float], Awaitable[None]]


class RuntimeStartError(RuntimeError):
    """Un rol no pudo iniciarse; el runtime revirtió los ya iniciados."""


class RouteViolationError(RuntimeError):
    """Un rol intentó publicar en un tópico que su ruta no permite."""


class WorkerState(StrEnum):
    """Estado del ciclo de vida de un rol."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Reintentos con backoff exponencial para fallos transitorios."""

    max_attempts: int = 3
    base_delay: float = 0.05
    backoff: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts debe ser al menos 1")

    def delay(self, attempt: int) -> float:
        """Espera tras el intento fallido número ``attempt`` (1 = primero)."""
        return self.base_delay * (self.backoff ** (attempt - 1))


class RoleHandler(Protocol):
    """Lógica de un rol: recibe metadata y devuelve la que debe publicarse."""

    async def start(self) -> None:
        """Prepara recursos del rol."""

    async def handle(
        self, topic: Topic, envelope: MetadataEnvelope
    ) -> Sequence[Outgoing]:
        """Procesa un mensaje; devuelve los mensajes a publicar."""

    async def stop(self) -> None:
        """Libera recursos del rol."""


HandlerSpec = RoleHandler | Callable[[str | None], RoleHandler]


class _SeenEvents:
    """Conjunto acotado (LRU) de claves de evento ya procesadas."""

    def __init__(self, capacity: int) -> None:
        self._items: OrderedDict[tuple[Topic, str], None] = OrderedDict()
        self._capacity = capacity

    def __contains__(self, key: tuple[Topic, str]) -> bool:
        if key in self._items:
            self._items.move_to_end(key)
            return True
        return False

    def add(self, key: tuple[Topic, str]) -> None:
        self._items[key] = None
        self._items.move_to_end(key)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)


def check_publish_allowed(route: AgentRoute, topic: Topic) -> None:
    """``ERRORS`` es transversal; el resto debe estar en ``route.publishes``."""
    if topic is Topic.ERRORS:
        return
    if topic not in route.publishes:
        permitidos = ", ".join(t.value for t in route.publishes) or "ninguno"
        raise RouteViolationError(
            f"El rol '{route.name}' no puede publicar en '{topic.value}' "
            f"(permitidos: {permitidos})"
        )


class RoleWorker:
    """Ejecuta un manejador: suscripciones, reintentos, deduplicación, salud."""

    def __init__(
        self,
        route: AgentRoute,
        handler: RoleHandler,
        hub: MetadataHub,
        *,
        instance: str | None = None,
        retry: RetryPolicy,
        sleep: Sleep,
        seen_capacity: int,
        shutdown_timeout: float,
    ) -> None:
        self.route = route
        self.handler = handler
        self.name = route.name if instance is None else f"{route.name}/{instance}"
        self.instance = instance
        self.state = WorkerState.CREATED
        self.processed = 0
        self.duplicates = 0
        self.failures = 0
        self.retries = 0
        self.last_error: str | None = None
        self._hub = hub
        self._retry = retry
        self._sleep = sleep
        self._seen = _SeenEvents(seen_capacity)
        self._shutdown_timeout = shutdown_timeout
        self._subscriptions: list[Any] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._busy = 0

    # -- ciclo de vida --

    async def start(self) -> None:
        self.state = WorkerState.STARTING
        try:
            await self.handler.start()
        except Exception:
            self.state = WorkerState.FAILED
            raise
        for topic in self.route.consumes:
            subscription = self._hub.subscribe(topic)
            self._subscriptions.append(subscription)
            self._tasks.append(
                asyncio.create_task(
                    self._consume(topic, subscription), name=f"{self.name}:{topic}"
                )
            )
        self.state = WorkerState.RUNNING

    async def stop(self) -> None:
        if self.state in (WorkerState.STOPPED, WorkerState.CREATED):
            return
        self.state = WorkerState.STOPPING
        for subscription in self._subscriptions:
            subscription.close()  # entrega lo encolado y luego termina
        if self._tasks:
            _, pending = await asyncio.wait(self._tasks, timeout=self._shutdown_timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                logger.warning("El rol '%s' no vació sus colas a tiempo", self.name)
        self._tasks.clear()
        self._subscriptions.clear()
        try:
            await self.handler.stop()
        finally:
            self.state = WorkerState.STOPPED

    # -- procesamiento --

    @property
    def idle(self) -> bool:
        return self._busy == 0 and all(s.pending == 0 for s in self._subscriptions)

    async def _consume(self, topic: Topic, subscription: Any) -> None:
        async for envelope in subscription:
            self._busy += 1
            try:
                await self._process(topic, envelope)
            finally:
                self._busy -= 1

    async def _process(self, topic: Topic, envelope: MetadataEnvelope) -> None:
        key = (topic, envelope.event_id)
        if key in self._seen:
            self.duplicates += 1
            return
        attempts = 0
        while True:
            attempts += 1
            try:
                outputs = await self.handler.handle(topic, envelope)
                await self._publish_outputs(outputs)
            except RouteViolationError as exc:  # error de programación: no reintentar
                await self._fail(topic, envelope, exc, attempts)
                return
            except Exception as exc:  # noqa: BLE001 - todo fallo del manejador se reintenta
                if attempts < self._retry.max_attempts:
                    self.retries += 1
                    await self._sleep(self._retry.delay(attempts))
                    continue
                await self._fail(topic, envelope, exc, attempts)
                return
            self._seen.add(key)  # solo tras éxito: un fallo permite reentrega
            self.processed += 1
            return

    async def _publish_outputs(self, outputs: Sequence[Outgoing]) -> None:
        for topic, _ in outputs:
            check_publish_allowed(self.route, parse_topic(topic))
        for topic, message in outputs:
            await self._hub.publish(topic, message)

    async def _fail(
        self,
        topic: Topic,
        envelope: MetadataEnvelope,
        error: Exception,
        attempts: int,
    ) -> None:
        self.failures += 1
        self.last_error = f"{type(error).__name__}: {error}"
        logger.error("Fallo en '%s' tras %d intentos: %s", self.name, attempts, error)
        report = build_error_envelope(
            self.name,
            error,
            failed=envelope,
            failed_topic=topic,
            stage="handle",
            attempts=attempts,
        )
        try:
            await self._hub.publish(Topic.ERRORS, report)
        except Exception:
            logger.exception(
                "No se pudo publicar el evento de error de '%s'", self.name
            )

    def health(self) -> dict[str, Any]:
        """Instantánea operativa: solo metadata, sin frames."""
        return {
            "role": self.route.name,
            "instance": self.instance,
            "state": self.state,
            "processed": self.processed,
            "duplicates": self.duplicates,
            "failures": self.failures,
            "retries": self.retries,
            "last_error": self.last_error,
        }


class AgentRuntime:
    """Instancia y gobierna los siete roles sobre un ``MetadataHub``."""

    def __init__(
        self,
        hub: MetadataHub,
        handlers: Mapping[str, HandlerSpec],
        *,
        routes: tuple[AgentRoute, ...] = MULTIAGENT_ROUTE,
        instances: Mapping[str, Sequence[str]] | None = None,
        retry: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
        heartbeat_interval: float | None = None,
        seen_capacity: int = 10_000,
        shutdown_timeout: float = 5.0,
    ) -> None:
        validate_route(routes)
        faltantes = [route.name for route in routes if route.name not in handlers]
        if faltantes:
            raise ValueError(f"Falta el manejador de los roles: {', '.join(faltantes)}")
        self._hub = hub
        self._routes = routes
        self._handlers = handlers
        self._instances = instances or {}
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._heartbeat_interval = heartbeat_interval
        self._seen_capacity = seen_capacity
        self._shutdown_timeout = shutdown_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._heartbeat: asyncio.Task[None] | None = None
        self._running = False
        self._built: dict[str, RoleHandler] = {}
        self._workers = self._build_workers()

    # -- construcción --

    def _build_workers(self) -> list[RoleWorker]:
        workers: list[RoleWorker] = []
        for route in self._routes:
            spec = self._handlers[route.name]
            names: Sequence[str | None] = self._instances.get(route.name) or [None]
            if len(names) > 1 and route.singleton:
                raise ValueError(
                    f"El rol '{route.name}' es único y no admite instancias"
                )
            if len(names) > 1 and hasattr(spec, "handle"):
                raise ValueError(
                    f"El rol '{route.name}' tiene varias instancias: se requiere una "
                    "fábrica de manejadores"
                )
            for instance in names:
                handler = (
                    spec if hasattr(spec, "handle") else spec(instance)  # type: ignore[operator]
                )
                self._built.setdefault(route.name, handler)  # type: ignore[arg-type]
                workers.append(
                    RoleWorker(
                        route,
                        handler,  # type: ignore[arg-type]
                        self._hub,
                        instance=instance,
                        retry=self._retry,
                        sleep=self._sleep,
                        seen_capacity=self._seen_capacity,
                        shutdown_timeout=self._shutdown_timeout,
                    )
                )
        return workers

    def handler(self, role: str) -> RoleHandler:
        """Manejador (primera instancia) de un rol, ya construido."""
        if role not in self._built:
            raise KeyError(f"Rol desconocido o aún no iniciado: {role}")
        return self._built[role]

    # -- ciclo de vida --

    async def start(self) -> None:
        """Inicia consumidores primero y productores al final."""
        if self._running:
            raise RuntimeError("El runtime ya está en ejecución")
        self._loop = asyncio.get_running_loop()
        started: list[RoleWorker] = []
        for worker in reversed(self._workers):
            try:
                await worker.start()
            except Exception as exc:
                for done in started:  # rollback en orden de flujo
                    await self._safe_stop(done)
                raise RuntimeStartError(
                    f"No se pudo iniciar el rol '{worker.name}': {exc}"
                ) from exc
            started.append(worker)
        self._running = True
        if self._heartbeat_interval is not None:
            self._heartbeat = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Detiene en orden de flujo para que cada etapa vacíe su cola."""
        if not self._running:
            return
        self._running = False
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
            self._heartbeat = None
        for worker in self._workers:
            await self._safe_stop(worker)

    @staticmethod
    async def _safe_stop(worker: RoleWorker) -> None:
        try:
            await worker.stop()
        except Exception:
            worker.state = WorkerState.FAILED
            logger.exception("Error al detener el rol '%s'", worker.name)

    # -- salud --

    def health(self) -> dict[str, dict[str, Any]]:
        """Salud por rol/instancia; únicamente metadata operativa."""
        return {worker.name: worker.health() for worker in self._workers}

    async def publish_health(self) -> None:
        """Publica la salud de cada rol en ``Topic.HEALTH``."""
        for worker in self._workers:
            await self._hub.publish(
                Topic.HEALTH,
                MetadataEnvelope(source=worker.name, payload=worker.health()),
            )

    async def _heartbeat_loop(self) -> None:
        assert self._heartbeat_interval is not None
        while True:
            await self.publish_health()
            await asyncio.sleep(self._heartbeat_interval)

    async def wait_idle(self, timeout: float = 5.0) -> None:
        """Espera a que no quede metadata en vuelo (útil en pruebas)."""
        deadline = time.monotonic() + timeout
        stable = 0
        while stable < 3:
            if all(worker.idle for worker in self._workers):
                stable += 1
            else:
                stable = 0
            if time.monotonic() > deadline:
                raise TimeoutError("El runtime no quedó inactivo a tiempo")
            await asyncio.sleep(0)

    # -- publicación desde fuera de los manejadores (pipeline, API) --

    def _route_of(self, role: str) -> AgentRoute:
        for route in self._routes:
            if route.name == role:
                return route
        raise KeyError(f"Rol desconocido: {role}")

    async def emit_envelope(
        self, role: str, topic: Topic, envelope: MetadataEnvelope
    ) -> MetadataEnvelope:
        """Publica un envelope en nombre de ``role`` respetando su ruta."""
        check_publish_allowed(self._route_of(role), parse_topic(topic))
        await self._hub.publish(topic, envelope)
        return envelope

    async def emit(
        self,
        role: str,
        topic: Topic,
        payload: Mapping[str, Any],
        *,
        stream_id: str | None = None,
        correlation_id: str | None = None,
        payload_version: int = 1,
    ) -> MetadataEnvelope:
        """Construye y publica metadata nueva en nombre de ``role``."""
        envelope = MetadataEnvelope(
            source=role,
            payload=payload,
            stream_id=stream_id,
            correlation_id=correlation_id,
            payload_version=payload_version,
        )
        return await self.emit_envelope(role, topic, envelope)

    def emit_threadsafe(
        self,
        role: str,
        topic: Topic,
        payload: Mapping[str, Any],
        *,
        stream_id: str | None = None,
        correlation_id: str | None = None,
        payload_version: int = 1,
    ) -> concurrent.futures.Future[MetadataEnvelope]:
        """Puente para callbacks del pipeline que corren en hilos propios.

        Solo viaja metadata; el frame permanece en el plano de video.
        """
        if self._loop is None or not self._running:
            raise RuntimeError("El runtime no está en ejecución")
        return asyncio.run_coroutine_threadsafe(
            self.emit(
                role,
                topic,
                payload,
                stream_id=stream_id,
                correlation_id=correlation_id,
                payload_version=payload_version,
            ),
            self._loop,
        )
