"""Vista de solo lectura del sistema que consume la API de observabilidad.

``SystemView`` es el adaptador que el system runner (#41) puede implementar;
``TapSystemView`` es la implementación por defecto sobre ``AgentRuntime`` (o
cualquier cosa que cumpla ``RuntimeProbe``) y un ``BusTap``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from bus import Topic

from .tap import BusTap

MessageListener = Callable[[Topic, dict[str, Any]], None]


class SystemView(Protocol):
    """Contrato de lectura que expone la API; todo es metadata JSON finita."""

    def agents(self) -> list[dict[str, Any]]:
        """Un dict por rol/instancia: name, role, instance, state, processed,
        duplicates, failures, retries, last_error, last_heartbeat, restarts,
        queue_depth."""

    def topics(self) -> list[dict[str, Any]]:
        """Un dict por tópico: topic, count, rate_per_s, window_s, drops,
        last_message_at."""

    def events(
        self, limit: int = 50, *, stream_id: str | None = None, zone: str | None = None
    ) -> list[dict[str, Any]]:
        """Últimos envelopes de ``events`` (más nuevo primero)."""

    def decisions(
        self, limit: int = 50, *, stream_id: str | None = None, zone: str | None = None
    ) -> list[dict[str, Any]]:
        """Últimas decisiones de fusión (envelopes con ``decision_id``)."""

    def listen(self, callback: MessageListener) -> Callable[[], None]:
        """Registra un oyente síncrono y no bloqueante de cada mensaje del bus.

        Devuelve la función para quitarlo. El callback no debe esperar nada.
        """


class RuntimeProbe(Protocol):
    """Lo que la vista necesita del ``AgentRuntime``."""

    def health(self) -> Mapping[str, Mapping[str, Any]]: ...

    def queue_depths(self) -> Mapping[str, int]: ...


class TapSystemView:
    """``SystemView`` sobre un ``BusTap`` y, opcionalmente, un runtime."""

    def __init__(self, tap: BusTap, runtime: RuntimeProbe | None = None) -> None:
        self._tap = tap
        self._runtime = runtime

    def bind_runtime(self, runtime: RuntimeProbe) -> None:
        self._runtime = runtime

    def agents(self) -> list[dict[str, Any]]:
        if self._runtime is not None:
            health: Mapping[str, Mapping[str, Any]] = self._runtime.health()
            depths: Mapping[str, int] = self._runtime.queue_depths()
        else:  # sin runtime: lo último que se vio por el bus
            health = self._tap.latest_health()
            depths = {}
        heartbeats = self._tap.heartbeats()
        restarts = self._tap.restarts()
        out: list[dict[str, Any]] = []
        for name, item in health.items():
            role = str(item.get("role", name.split("/")[0]))
            out.append(
                {
                    "name": name,
                    "role": role,
                    "instance": item.get("instance"),
                    "state": str(item.get("state")),
                    "processed": item.get("processed", 0),
                    "duplicates": item.get("duplicates", 0),
                    "failures": item.get("failures", 0),
                    "retries": item.get("retries", 0),
                    "last_error": item.get("last_error"),
                    "last_heartbeat": heartbeats.get(name),
                    "restarts": restarts.get(role, 0),
                    "queue_depth": depths.get(name),
                }
            )
        return out

    def topics(self) -> list[dict[str, Any]]:
        return self._tap.topic_stats()

    def events(
        self, limit: int = 50, *, stream_id: str | None = None, zone: str | None = None
    ) -> list[dict[str, Any]]:
        return self._tap.events(limit, stream_id=stream_id, zone=zone)

    def decisions(
        self, limit: int = 50, *, stream_id: str | None = None, zone: str | None = None
    ) -> list[dict[str, Any]]:
        return self._tap.decisions(limit, stream_id=stream_id, zone=zone)

    def listen(self, callback: MessageListener) -> Callable[[], None]:
        return self._tap.add_listener(callback)
