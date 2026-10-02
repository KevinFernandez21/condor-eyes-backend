"""Estado en memoria del dashboard: lo último que dijo la API y el WebSocket.

Todo ocurre en el bucle de asyncio del servidor del dashboard; no hay hilos.
``snapshot`` devuelve el JSON que consume la página.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from . import viewmodel as vm

CAMERA_PLACEHOLDER = {
    "available": False,
    "message": (
        "La vista previa de cámara vive del lado del pipeline y aún no está "
        "implementada. La API no transporta frames."
    ),
}


class DashboardState:
    def __init__(
        self,
        *,
        feed_size: int = 300,
        alert_size: int = 100,
        site: Mapping[str, Any] | None = None,
        routes: Sequence[vm.RouteSpec] | None = None,
    ) -> None:
        self._site = site
        self._routes = list(routes) if routes is not None else None
        self._feed: deque[dict[str, Any]] = deque(maxlen=feed_size)  # más nuevo primero
        self._decisions: deque[dict[str, Any]] = deque(maxlen=alert_size)
        self._locations: deque[dict[str, Any]] = deque(maxlen=200)
        self._streams: deque[dict[str, Any]] = deque(maxlen=200)
        self._seen_decisions: OrderedDict[str, None] = OrderedDict()  # índice acotado
        self._agents: list[dict[str, Any]] = []
        self._topics: list[dict[str, Any]] = []
        self._api_health: dict[str, Any] | None = None
        self._api_ok = False
        self._api_error: str | None = "sin datos todavía"
        self._api_last_ok: str | None = None
        self._ws_ok = False
        self._ws_error: str | None = None
        self._ws_reconnects = 0
        self._lag_total = 0
        self._lag_events = 0
        self._hellos = 0

    # -- entradas -----------------------------------------------------------

    def on_ws_message(self, message: Any) -> None:
        if not isinstance(message, Mapping):
            return
        kind = message.get("type")
        if kind == "hello":
            if self._hellos:
                self._ws_reconnects += 1
            self._hellos += 1
            self._ws_ok = True
            self._ws_error = None
        elif kind == "lag":
            self._lag_events += 1
            total = message.get("dropped_total")
            if isinstance(total, int):
                self._lag_total = total
            else:
                self._lag_total += int(message.get("dropped") or 0)
        elif kind == "envelope":
            data = message.get("data")
            if isinstance(data, Mapping):
                self._ingest(data)

    def on_ws_closed(self, reason: str) -> None:
        self._ws_ok = False
        self._ws_error = reason

    def on_poll(self, data: Mapping[str, Any]) -> None:
        self._agents = list(data.get("agents") or [])
        self._topics = list(data.get("topics") or [])
        self._api_health = dict(data.get("health") or {})
        self._api_ok = True
        self._api_error = None
        self._api_last_ok = datetime.now(UTC).isoformat()
        for env in data.get("decisions") or []:  # vienen del más nuevo al más antiguo
            if isinstance(env, Mapping):
                self._add_decision(env)

    def on_poll_error(self, reason: str) -> None:
        self._api_ok = False
        self._api_error = reason

    def _ingest(self, env: Mapping[str, Any]) -> None:
        self._feed.appendleft(vm.feed_entry(env))
        if vm.is_decision(env):
            self._add_decision(env)
        if vm.is_location(env):
            self._locations.appendleft(_slim(env))
        if env.get("topic") in ("system.health", "stream.status"):
            self._streams.appendleft(_slim(env))

    def _add_decision(self, env: Mapping[str, Any]) -> None:
        key = env.get("event_id") or (env.get("payload") or {}).get("decision_id")
        if key is not None:  # sin ningún id no se deduplica: nunca se descarta una decisión
            key = str(key)
            if key in self._seen_decisions:
                return
            self._seen_decisions[key] = None
            while len(self._seen_decisions) > (self._decisions.maxlen or 0):
                self._seen_decisions.popitem(last=False)
        self._decisions.appendleft(_slim(env))

    # -- salida -------------------------------------------------------------

    def snapshot(
        self,
        *,
        topics: Iterable[str] | None = None,
        text: str | None = None,
        correlation_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        graph = vm.build_graph(self._agents, self._topics, self._routes)
        graph["stale"] = not self._api_ok
        decisions = sorted(
            self._decisions, key=lambda e: str(e.get("created_at") or ""), reverse=True
        )
        streams = vm.stream_health(self._streams)
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "api": {
                "connected": self._api_ok,
                "error": self._api_error,
                "last_ok": self._api_last_ok,
            },
            "ws": {
                "connected": self._ws_ok,
                "error": self._ws_error,
                "reconnects": self._ws_reconnects,
            },
            "lag": {"dropped_total": self._lag_total, "events": self._lag_events},
            "graph": graph,
            "feed": vm.filter_feed(
                self._feed, topics=topics, text=text, correlation_id=correlation_id, limit=limit
            ),
            "feed_topics": sorted({str(t.get("topic")) for t in self._topics}),
            "alerts": vm.format_alerts(decisions, limit=30),
            "site": vm.site_view(self._site, self._locations),
            "tags": vm.tag_presence(self._locations),
            "health": vm.health_strip(self._api_health, streams, connected=self._api_ok),
            "camera": dict(CAMERA_PLACEHOLDER),
        }


def _slim(env: Mapping[str, Any]) -> dict[str, Any]:
    """Copia del envelope con el payload ya redactado."""
    out = dict(env)
    out["payload"] = vm.redact(env.get("payload") or {})
    return out
