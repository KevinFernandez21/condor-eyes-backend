"""Conexión del ejecutor con la API de observabilidad (#42).

- ``RunnerView``: ``TapSystemView`` que, además de los roles del runtime,
  muestra los componentes del ejecutor (cámara, detector, fusión, ubicación...)
  como entradas ``component/<nombre>`` con ``role: "component"``, de modo que el
  dashboard ve la degradación sin una ruta nueva en la API.
- ``RunnerCommsHandler``: manejador del rol ``comms``; su ciclo de vida arranca
  el tap del bus y, si está activada, la API HTTP/WS (uvicorn, diferido).
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Mapping
from typing import Any

from agents.handlers import AlertSink, CommsHandler
from comms import BusTap, TapSystemView
from comms.server import CommsServer

ComponentSource = Callable[[], Mapping[str, Mapping[str, Any]]]


def ensure_port_free(host: str, port: int) -> None:
    """Falla con un mensaje claro si el puerto ya está en uso.

    uvicorn terminaría el proceso con ``sys.exit`` ante un bind fallido; mejor
    detectarlo antes y dejar que el ejecutor apague limpio.
    """
    if port == 0:
        return
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(
                f"El puerto {port} de {host} ya está en uso ({exc.strerror}); "
                "use --api-port con otro puerto."
            ) from exc


class RunnerView(TapSystemView):
    """Vista del tap con los componentes del ejecutor añadidos a ``agents()``."""

    def __init__(self, tap: BusTap, components: ComponentSource) -> None:
        super().__init__(tap)
        self._components = components

    def agents(self) -> list[dict[str, Any]]:
        out = super().agents()
        for name, health in self._components().items():
            status = str(health.get("status", "unknown"))
            out.append(
                {
                    "name": f"component/{name}",
                    "role": "component",
                    "instance": name,
                    "state": status,
                    "processed": int(health.get("published", 0) or 0),
                    "duplicates": 0,
                    "failures": int(health.get("errors", 0) or 0),
                    "retries": 0,
                    "last_error": (
                        str(health.get("detail") or "") or None
                        if status == "degraded"
                        else None
                    ),
                    "last_heartbeat": None,
                    "restarts": 0,
                    "queue_depth": None,
                }
            )
        return out


class RunnerCommsHandler(CommsHandler):
    """Rol ``comms``: tap del bus (siempre) y servidor de la API (opcional)."""

    def __init__(
        self,
        sink: AlertSink,
        tap: BusTap,
        server: CommsServer | None,
        bind: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(sink)
        self._tap = tap
        self._server = server
        self._bind = bind

    async def start(self) -> None:
        if self._server is not None and self._bind is not None:
            ensure_port_free(*self._bind)
        await self._tap.start()  # antes que los productores
        if self._server is not None:
            try:
                await self._server.start()
            except BaseException:
                await self._tap.stop()
                raise

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.stop()
        await self._tap.stop()
