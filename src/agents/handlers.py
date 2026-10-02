"""Manejadores de los siete roles: reciben metadata y devuelven metadata.

Cada manejador es lógica de dominio pura sobre ``MetadataEnvelope``; no conoce
el transporte (``bus``) ni el ciclo de vida (``agents.runtime``). Los efectos
externos (guardar clips, enviar alertas, mover cámaras) entran por puertos
inyectables (``EventSink``, ``AlertSink``, callbacks) para poder probarlos y
sustituirlos sin tocar el runtime.

Convención de ``event_id`` derivado: ``<padre>/<sufijo>``. Es determinista, de
modo que reprocesar el mismo mensaje de entrada produce los mismos
identificadores de salida y los consumidores pueden deduplicar.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from bus import MetadataEnvelope, Topic

from .runtime import HandlerSpec, Outgoing, WorkerState

EventRule = Callable[[MetadataEnvelope], Iterable[Mapping[str, Any]]]
"""Regla del rol ``event``: dado un envelope de tracks, produce payloads de evento."""

TrackFn = Callable[[MetadataEnvelope], Mapping[str, Any]]
"""Función de seguimiento: convierte un envelope de detecciones en payload de tracks."""

CommandCallback = Callable[[MetadataEnvelope], Awaitable[None]]


class EventSink(Protocol):
    """Puerto de persistencia de eventos (clips + SQLite en producción)."""

    async def save(self, envelope: MetadataEnvelope) -> None:
        """Persiste un evento."""


class AlertSink(Protocol):
    """Puerto de salida hacia el exterior (API HTTP/WS en producción)."""

    async def send(self, envelope: MetadataEnvelope) -> None:
        """Entrega una alerta o evento al exterior."""


class _BaseHandler:
    async def start(self) -> None:
        """Sin recursos que preparar por defecto."""

    async def stop(self) -> None:
        """Sin recursos que liberar por defecto."""


class IngestHandler(_BaseHandler):
    """Un handler por cámara: atiende comandos dirigidos a su stream."""

    def __init__(
        self,
        stream_id: str | None = None,
        on_command: CommandCallback | None = None,
    ) -> None:
        self.stream_id = stream_id
        self._on_command = on_command

    async def handle(
        self, topic: Topic, envelope: MetadataEnvelope
    ) -> Sequence[Outgoing]:
        if topic is not Topic.COMMANDS:
            return []
        if self.stream_id is not None and envelope.stream_id != self.stream_id:
            return []  # el comando es para otra cámara
        if self._on_command is not None:
            await self._on_command(envelope)
        return []


class InferenceHandler(_BaseHandler):
    """Lleva la cuenta de streams activos; las detecciones llegan del pipeline."""

    def __init__(self) -> None:
        self.active_streams: set[str] = set()

    async def handle(
        self, topic: Topic, envelope: MetadataEnvelope
    ) -> Sequence[Outgoing]:
        if topic is Topic.STREAM_STATUS and envelope.stream_id is not None:
            if envelope.payload.get("state") == "up":
                self.active_streams.add(envelope.stream_id)
            else:
                self.active_streams.discard(envelope.stream_id)
        return []


def _passthrough_tracks(envelope: MetadataEnvelope) -> Mapping[str, Any]:
    return {"tracks": list(envelope.payload.get("detections", []))}


class TrackerHandler(_BaseHandler):
    """Detecciones -> tracks. El algoritmo (NvDCF/ByteTrack) es inyectable."""

    def __init__(self, track_fn: TrackFn | None = None) -> None:
        self._track_fn = track_fn or _passthrough_tracks

    async def handle(
        self, topic: Topic, envelope: MetadataEnvelope
    ) -> Sequence[Outgoing]:
        payload = self._track_fn(envelope)
        return [(Topic.TRACKS, envelope.derive("tracker", payload, suffix="tracks"))]


class EventHandler(_BaseHandler):
    """Tracks -> eventos mediante reglas inyectables (zonas, merodeo, conteo)."""

    def __init__(self, rules: Iterable[EventRule] = ()) -> None:
        self._rules = tuple(rules)

    async def handle(
        self, topic: Topic, envelope: MetadataEnvelope
    ) -> Sequence[Outgoing]:
        outputs: list[Outgoing] = []
        index = 0
        for rule in self._rules:
            for payload in rule(envelope):
                outputs.append(
                    (
                        Topic.EVENTS,
                        envelope.derive("event", payload, suffix=f"events/{index}"),
                    )
                )
                index += 1
        return outputs


class StorageHandler(_BaseHandler):
    """Eventos -> persistencia."""

    def __init__(self, sink: EventSink) -> None:
        self._sink = sink

    async def handle(
        self, topic: Topic, envelope: MetadataEnvelope
    ) -> Sequence[Outgoing]:
        await self._sink.save(envelope)
        return []


class CommsHandler(_BaseHandler):
    """Puente hacia el exterior: eventos como alertas y última salud conocida."""

    def __init__(self, sink: AlertSink) -> None:
        self._sink = sink
        self.latest_health: dict[str, Mapping[str, Any]] = {}

    async def handle(
        self, topic: Topic, envelope: MetadataEnvelope
    ) -> Sequence[Outgoing]:
        if topic is Topic.EVENTS:
            await self._sink.send(envelope)
        elif topic is Topic.HEALTH:
            self.latest_health[envelope.source] = envelope.payload
        return []


class SupervisorHandler(_BaseHandler):
    """Observa salud, estado de streams y errores; solicita reinicios.

    Solo consume metadata operativa: no tiene acceso a frames porque ningún
    tópico del bus los transporta.
    """

    def __init__(self) -> None:
        self.health: dict[str, Mapping[str, Any]] = {}
        self.streams: dict[str, str] = {}
        self.errors: list[Mapping[str, Any]] = []

    async def handle(
        self, topic: Topic, envelope: MetadataEnvelope
    ) -> Sequence[Outgoing]:
        if topic is Topic.STREAM_STATUS:
            if envelope.stream_id is not None:
                self.streams[envelope.stream_id] = str(envelope.payload.get("state"))
        elif topic is Topic.ERRORS:
            self.errors.append(envelope.payload)
        elif topic is Topic.HEALTH:
            self.health[envelope.source] = envelope.payload
            if envelope.payload.get("state") == WorkerState.FAILED.value:
                target = envelope.payload.get("role", envelope.source)
                command = envelope.derive(
                    "supervisor",
                    {"action": "restart", "target": target},
                    suffix="commands",
                )
                return [(Topic.COMMANDS, command)]
        return []

    def report(self) -> dict[str, Any]:
        """Resumen de salud para la API o para el agente ReAct."""
        return {
            "health": dict(self.health),
            "streams": dict(self.streams),
            "errors": len(self.errors),
        }


def build_default_handlers(
    *,
    storage_sink: EventSink,
    alert_sink: AlertSink,
    event_rules: Iterable[EventRule] = (),
    track_fn: TrackFn | None = None,
) -> dict[str, HandlerSpec]:
    """Manejadores de los siete roles de la ruta canónica."""
    return {
        "ingest": IngestHandler(),
        "inference": InferenceHandler(),
        "tracker": TrackerHandler(track_fn),
        "event": EventHandler(event_rules),
        "storage": StorageHandler(storage_sink),
        "supervisor": SupervisorHandler(),
        "comms": CommsHandler(alert_sink),
    }
