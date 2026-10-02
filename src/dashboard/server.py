"""Servidor del dashboard: página estática + ``/api/state`` para la UI.

La página (HTML/CSS/JS sin dependencias) consulta ``/api/state`` de este mismo
servidor; el servidor es quien habla con la API de observabilidad, así el token
no llega al navegador ni hace falta CORS.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import viewmodel as vm
from .client import poll_loop, ws_loop
from .config import DashboardConfig
from .state import DashboardState

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class DashboardConfigurationError(ValueError):
    """Configuración insegura o inválida del servidor del dashboard."""


def is_local_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_bind(host: str, allow_lan: bool) -> None:
    """El dashboard no tiene login propio: fuera de loopback exige confirmación."""
    if not is_local_host(host) and not allow_lan:
        raise DashboardConfigurationError(
            f"El dashboard no tiene autenticación propia; escuchar en {host!r} lo expone "
            "a la red. Usa 127.0.0.1 o confirma con --allow-lan."
        )


def build_state(config: DashboardConfig) -> DashboardState:
    site = None
    try:
        site = vm.load_site_map(config.site_map_path)
    except ValueError as exc:
        logger.warning("Se ignora el mapa de sitio: %s", exc)
    return DashboardState(feed_size=config.feed_size, site=site)


def create_app(
    config: DashboardConfig | None = None,
    *,
    state: DashboardState | None = None,
    start_clients: bool = True,
) -> FastAPI:
    """App del dashboard. ``start_clients=False`` no abre conexiones (pruebas)."""
    config = config or DashboardConfig.from_env()
    state = state or build_state(config)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        tasks: list[asyncio.Task[None]] = []
        if start_clients:
            tasks = [
                asyncio.create_task(poll_loop(config, state), name="dashboard-poll"),
                asyncio.create_task(ws_loop(config, state), name="dashboard-ws"),
            ]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="Condor Eye - Dashboard del sistema", lifespan=lifespan)

    @app.get("/api/state")
    def api_state(
        topics: str | None = Query(None, description="Tópicos del feed, separados por coma"),
        q: str | None = Query(None, max_length=100),
        correlation_id: str | None = Query(None, max_length=100),
        limit: int = Query(100, ge=1, le=300),
    ) -> JSONResponse:
        wanted = [t.strip() for t in topics.split(",") if t.strip()] if topics else None
        body: dict[str, Any] = state.snapshot(
            topics=wanted, text=q, correlation_id=correlation_id, limit=limit
        )
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.dashboard = state
    return app
