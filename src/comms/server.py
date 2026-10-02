"""Servidor HTTP/WS de la API y su manejador para el rol ``comms``.

``uvicorn`` se importa de forma diferida dentro de ``start``: importar el
paquete no exige tenerlo cargado.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from typing import Any

from agents.handlers import AlertSink, CommsHandler
from bus import MetadataEnvelope, MetadataHub
from tracing import LangfuseDecisionTracer, TraceStore, create_tracer_from_env

from .api import DEFAULT_WS_QUEUE, check_bind, create_app
from .tap import BusTap
from .view import RuntimeProbe, TapSystemView


class _NullSink:
    async def send(self, envelope: MetadataEnvelope) -> None:
        """La API de observabilidad no reenvía alertas; solo las expone."""


class CommsServer:
    """Ejecuta la app ASGI con uvicorn dentro del loop del runtime."""

    def __init__(self, app: Any, host: str, port: int) -> None:
        self._app = app
        self._host = host
        self._requested_port = port
        self._server: Any = None
        self._task: asyncio.Task[None] | None = None

    @property
    def port(self) -> int:
        """Puerto real (útil con ``port=0``)."""
        if self._server is not None and self._server.servers:
            sockets = self._server.servers[0].sockets
            if sockets:
                return int(sockets[0].getsockname()[1])
        return self._requested_port

    async def start(self, timeout: float = 10.0) -> None:
        import uvicorn  # diferido: solo se necesita al servir

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._requested_port,
            log_level="warning",
            timeout_graceful_shutdown=2,
        )
        server = uvicorn.Server(config)
        server.capture_signals = contextlib.nullcontext  # type: ignore[method-assign,assignment]  # las señales las gobierna el runtime
        self._server = server
        self._task = asyncio.create_task(server.serve(), name="comms-api")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not server.started:
            if self._task.done():
                error = self._task.exception()
                raise RuntimeError(f"La API de observabilidad no pudo arrancar: {error}")
            if loop.time() > deadline:
                await self.stop()
                raise TimeoutError("La API de observabilidad no arrancó a tiempo")
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        if self._server is None or self._task is None:
            return
        self._server.should_exit = True
        task, self._task = self._task, None
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        except Exception:  # noqa: BLE001, S110 - ya se está parando
            pass


class ObservabilityCommsHandler(CommsHandler):
    """Manejador del rol ``comms``: tap del bus + API de observabilidad.

    El runtime lo arranca y lo detiene como a cualquier rol. La ruta de
    ``comms`` (consume ``events`` y ``system.health``) se conserva, de modo que
    un ``AlertSink`` opcional sigue recibiendo las alertas.
    """

    def __init__(
        self,
        hub: MetadataHub,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        token: str | None = None,
        sink: AlertSink | None = None,
        ws_queue_size: int = DEFAULT_WS_QUEUE,
        allowed_origins: Sequence[str] | None = None,
        tap: BusTap | None = None,
        traces: TraceStore | None = None,
        langfuse: bool = True,
    ) -> None:
        check_bind(host, token)  # falla antes de arrancar nada
        super().__init__(sink or _NullSink())
        self.tap = tap or BusTap(hub)
        self.traces = traces or TraceStore()
        self.tap.add_listener(self.traces.record)  # antes que Langfuse: la cadena ya incluye el mensaje
        self._use_langfuse = langfuse
        self._tracer: LangfuseDecisionTracer | None = None
        self._remove_tracer: Callable[[], None] | None = None
        self.view = TapSystemView(self.tap)
        self.app = create_app(
            self.view,
            token=token,
            host=host,
            ws_queue_size=ws_queue_size,
            allowed_origins=allowed_origins,
            traces=self.traces,
        )
        self._server = CommsServer(self.app, host, port)

    @property
    def port(self) -> int:
        return self._server.port

    def bind_runtime(self, runtime: RuntimeProbe) -> None:
        """Conecta el runtime para exponer estado y profundidad de cola."""
        self.view.bind_runtime(runtime)

    async def start(self) -> None:
        if self._use_langfuse and self._tracer is None:
            # Solo si el entorno lo configura (claves + host); si no, no hace nada.
            self._tracer = create_tracer_from_env(self.traces)
            if self._tracer is not None:
                self._remove_tracer = self.tap.add_listener(self._tracer.on_message)
        await self.tap.start()
        try:
            await self._server.start()
        except BaseException:
            await self.tap.stop()
            self._close_tracer()
            raise

    async def stop(self) -> None:
        await self._server.stop()
        await self.tap.stop()
        self._close_tracer()

    def _close_tracer(self) -> None:
        if self._remove_tracer is not None:
            self._remove_tracer()
            self._remove_tracer = None
        if self._tracer is not None:
            self._tracer.close()
            self._tracer = None
