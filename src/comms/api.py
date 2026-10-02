"""API de observabilidad de solo lectura (FastAPI) del rol ``comms``.

Expone agentes, tópicos, eventos y decisiones por HTTP y el flujo vivo del bus
por WebSocket. Solo devuelve metadata JSON finita ya validada por el bus; los
identificadores de tags y personas se muestran tal como viajan (seudonimizados)
y esta capa no tiene forma de revertirlos. No sirve video: la vista previa
pertenece al lado del pipeline.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
)
from starlette.websockets import WebSocketDisconnect

from bus import InvalidTopicError, Topic, parse_topic

from .tap import payload_stream
from .view import SystemView

logger = logging.getLogger(__name__)

MAX_LIMIT = 500
DEFAULT_WS_QUEUE = 256


class ConfigurationError(ValueError):
    """Configuración insegura o inválida de la API."""


def is_local_host(host: str) -> bool:
    """True si ``host`` es loopback (127.0.0.0/8, ::1 o ``localhost``)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def check_bind(host: str, token: str | None) -> None:
    """Rechaza un bind no local sin token."""
    if not is_local_host(host) and not token:
        raise ConfigurationError(
            f"Bind no local ('{host}') sin token: la API de observabilidad se "
            "niega a arrancar. Define un token o usa 127.0.0.1."
        )


class ClientChannel:
    """Cola acotada por cliente WebSocket: descarta el más antiguo, nunca bloquea.

    ``push`` es síncrono y O(1): lo invoca el tap dentro del camino del bus.
    Solo ``run`` espera (al socket), y únicamente bloquea a su propio cliente.
    """

    def __init__(self, max_queue: int = DEFAULT_WS_QUEUE) -> None:
        if max_queue < 1:
            raise ValueError("max_queue debe ser al menos 1")
        self._queue: deque[dict[str, Any]] = deque(maxlen=max_queue)
        self._event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._unreported = 0
        self.dropped = 0
        self.closed = False

    @property
    def queued(self) -> int:
        return len(self._queue)

    def push(self, data: dict[str, Any]) -> None:
        if self.closed:
            return
        if len(self._queue) == self._queue.maxlen:
            self.dropped += 1
            self._unreported += 1
        self._queue.append(data)  # con maxlen descarta el más antiguo
        self._wake()

    def close(self) -> None:
        self.closed = True
        self._wake()

    def _wake(self) -> None:
        loop = self._loop
        if loop is None:
            return  # run() revisará la cola al arrancar
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._event.set()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(self._event.set)

    async def run(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Envía al cliente hasta cerrar; un fallo de envío cierra el canal."""
        self._loop = asyncio.get_running_loop()
        try:
            while True:
                if self._unreported:
                    lag = {
                        "type": "lag",
                        "dropped": self._unreported,
                        "dropped_total": self.dropped,
                    }
                    self._unreported = 0
                    await send(lag)
                    continue
                if self._queue:
                    await send({"type": "envelope", "data": self._queue.popleft()})
                    continue
                if self.closed:
                    return
                self._event.clear()
                if self._queue or self._unreported or self.closed:
                    continue
                await self._event.wait()
        except Exception:  # noqa: BLE001 - socket muerto o roto: se cierra este cliente
            logger.info("Cliente WebSocket desconectado; se libera su canal")
        finally:
            self.closed = True
            self._queue.clear()
            if on_close is not None:
                on_close()


class _State:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.clients: set[ClientChannel] = set()
        self.dropped_closed = 0


def create_app(
    view: SystemView,
    *,
    token: str | None = None,
    host: str = "127.0.0.1",
    ws_queue_size: int = DEFAULT_WS_QUEUE,
) -> FastAPI:
    """Construye la app. Falla si ``host`` no es local y no hay ``token``."""
    check_bind(host, token)
    token = token or None
    state = _State()

    def authorized(headers: Any, query_token: str | None) -> bool:
        if token is None:
            return True
        candidates = [headers.get("x-api-token")]
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            candidates.append(auth[7:].strip())
        candidates.append(query_token)
        return any(c is not None and hmac.compare_digest(c, token) for c in candidates)

    def require_token(request: Request) -> None:
        # En HTTP el token solo va por cabecera (la URL acaba en logs).
        if not authorized(request.headers, None):
            raise HTTPException(status_code=401, detail="Token ausente o inválido")

    app = FastAPI(
        title="Condor Eye - API de observabilidad",
        description="Solo lectura: metadata del bus, agentes, eventos y decisiones.",
        version="1",
    )
    router = APIRouter(dependencies=[Depends(require_token)])  # solo rutas HTTP

    Limit = Query(50, ge=1, le=MAX_LIMIT, description="Máximo de elementos")

    @router.get("/health")
    def health() -> dict[str, Any]:
        agents = view.agents()
        failed = sum(1 for a in agents if a["state"] == "failed")
        running = sum(1 for a in agents if a["state"] == "running")
        return {
            "status": "degraded" if failed else "ok",
            "agents_total": len(agents),
            "agents_running": running,
            "agents_failed": failed,
            "uptime_s": round(time.monotonic() - state.started, 3),
            "ws_clients": len(state.clients),
            "ws_dropped_total": state.dropped_closed
            + sum(c.dropped for c in state.clients),
        }

    @router.get("/agents")
    def agents() -> dict[str, Any]:
        return {"agents": view.agents()}

    @router.get("/topics")
    def topics() -> dict[str, Any]:
        return {"topics": view.topics()}

    @router.get("/events")
    def events(
        limit: int = Limit, stream_id: str | None = None, zone: str | None = None
    ) -> dict[str, Any]:
        return {"events": view.events(limit, stream_id=stream_id, zone=zone)}

    @router.get("/decisions")
    def decisions(
        limit: int = Limit, stream_id: str | None = None, zone: str | None = None
    ) -> dict[str, Any]:
        return {"decisions": view.decisions(limit, stream_id=stream_id, zone=zone)}

    app.include_router(router)

    @app.websocket("/ws")
    async def ws_endpoint(
        websocket: WebSocket,
        topics: str | None = None,
        stream_id: str | None = None,
        token: str | None = None,
    ) -> None:
        if not authorized(websocket.headers, token):
            await websocket.close(code=1008, reason="Token ausente o inválido")
            return
        try:
            wanted = (
                frozenset(parse_topic(t.strip()) for t in topics.split(",") if t.strip())
                if topics
                else frozenset(Topic)
            )
        except InvalidTopicError as exc:
            await websocket.close(code=1008, reason=str(exc)[:120])
            return
        await websocket.accept()
        channel = ClientChannel(ws_queue_size)
        state.clients.add(channel)

        def on_message(topic: Topic, data: dict[str, Any]) -> None:
            if topic not in wanted:
                return
            if stream_id is not None and payload_stream(data) != stream_id:
                return
            channel.push(data)

        remove = view.listen(on_message)
        sender: asyncio.Task[None] | None = None
        receiver: asyncio.Task[Any] | None = None
        try:
            await websocket.send_json(
                {
                    "type": "hello",
                    "topics": sorted(t.value for t in wanted),
                    "stream_id": stream_id,
                    "queue_size": ws_queue_size,
                }
            )
            sender = asyncio.create_task(channel.run(websocket.send_json))
            receiver = asyncio.create_task(_drain_incoming(websocket))
            await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.info("WebSocket terminó con error", exc_info=True)
        finally:
            remove()
            channel.close()
            for task in (sender, receiver):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(t for t in (sender, receiver) if t is not None), return_exceptions=True
            )
            state.clients.discard(channel)
            state.dropped_closed += channel.dropped
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001, S110 - ya cerrado por el cliente
                pass

    return app


async def _drain_incoming(websocket: WebSocket) -> None:
    """Ignora lo que envíe el cliente; solo sirve para detectar la desconexión."""
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return
