"""Cliente de la API de observabilidad (#42): HTTP periódico + WebSocket con backoff.

Es la única parte del dashboard que habla por red. Nunca toca el bus ni el
runtime: todo lo que sabe viene de ``/health``, ``/agents``, ``/topics``,
``/decisions`` y ``/ws``. Ningún fallo aquí se propaga a la UI: se registra en
el estado y se reintenta con espera creciente.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .config import DashboardConfig
from .state import DashboardState

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Backoff:
    """Espera exponencial acotada; ``reset`` tras una conexión exitosa."""

    initial: float = 1.0
    factor: float = 2.0
    maximum: float = 15.0
    _attempt: int = 0

    def next(self) -> float:
        delay = min(self.initial * self.factor**self._attempt, self.maximum)
        self._attempt += 1
        return delay

    def reset(self) -> None:
        self._attempt = 0


def _http_client(config: DashboardConfig) -> httpx.AsyncClient:
    # verify=False solo para http:// (no hay TLS que verificar); evita construir
    # un contexto SSL que en algunos Windows falla al leer el almacén de certificados.
    plain = config.api_url.startswith("http://")
    return httpx.AsyncClient(
        base_url=config.api_url,
        headers=config.headers(),
        timeout=httpx.Timeout(3.0),
        verify=not plain,
    )


async def poll_once(client: httpx.AsyncClient) -> dict[str, Any]:
    """Una ronda de lecturas HTTP; lanza si la API no responde bien."""
    health, agents, topics, decisions = await asyncio.gather(
        client.get("/health"),
        client.get("/agents"),
        client.get("/topics"),
        client.get("/decisions", params={"limit": 50}),
    )
    for response in (health, agents, topics, decisions):
        response.raise_for_status()
    return {
        "health": health.json(),
        "agents": agents.json().get("agents", []),
        "topics": topics.json().get("topics", []),
        "decisions": decisions.json().get("decisions", []),
    }


def describe_error(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return "token ausente o inválido (401)" if code == 401 else f"la API respondió {code}"
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return "la API no responde (tiempo agotado)"
    if isinstance(exc, (httpx.ConnectError, OSError)):
        return "no se puede conectar con la API"
    rcvd = getattr(exc, "rcvd", None)
    if getattr(rcvd, "code", None) == 1008:
        return "la API rechazó el WebSocket (token inválido, 1008)"
    return f"error: {type(exc).__name__}"


async def poll_loop(config: DashboardConfig, state: DashboardState) -> None:
    async with _http_client(config) as client:
        while True:
            try:
                state.on_poll(await poll_once(client))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - la UI debe sobrevivir a todo
                state.on_poll_error(describe_error(exc))
            await asyncio.sleep(config.poll_interval_s)


async def ws_loop(
    config: DashboardConfig, state: DashboardState, backoff: Backoff | None = None
) -> None:
    """Sigue ``/ws`` para siempre; reconecta con backoff y nunca bloquea."""
    from websockets.asyncio.client import connect  # diferido: no pesa al importar

    backoff = backoff or Backoff()
    while True:
        try:
            async with connect(
                config.ws_url(), open_timeout=5, ping_interval=20, max_size=2**21
            ) as ws:
                async for raw in ws:
                    try:
                        message = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(message, dict) and message.get("type") == "hello":
                        backoff.reset()
                    state.on_ws_message(message)
                state.on_ws_closed("conexión cerrada por la API")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # websockets no incluye la URL (con token) en el mensaje de estos errores.
            state.on_ws_closed(describe_error(exc))
        delay = backoff.next()
        logger.info("WS desconectado; reintento en %.1f s", delay)
        await asyncio.sleep(delay)
