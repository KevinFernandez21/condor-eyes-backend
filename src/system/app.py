"""``SystemApp``: ensambla y ejecuta todo el sistema multiagente en un proceso.

Plano de agentes: ``AgentRuntime`` con los siete roles sobre un
``InstrumentedHub`` (``InMemoryHub``). Plano de video: ``LiveVideoPipeline`` en
hilos propios; su único puente al bus es ``MetadataPublisher`` y el detector
vive del lado del pipeline (``processor``), así que **ningún frame llega al
bus**. Fusión y simuladores son componentes que hablan con el hub tipado.

API pública para otros componentes (observabilidad, dashboard):

- ``SystemApp.runtime`` (``AgentRuntime``), ``SystemApp.hub`` (``InMemoryHub``)
- ``SystemApp.snapshot()``: estado JSON estricto (agentes, tópicos, componentes)
- ``SystemApp.start()`` / ``stop()`` / ``run(duration=, stop_event=)``
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, Protocol

from agents import AgentRuntime
from agents.handlers import build_default_handlers
from bus import InMemoryHub, MetadataEnvelope, Topic
from comms import BusTap
from comms.api import check_bind, create_app
from comms.server import CommsServer
from compare.detectors import Detector
from pipeline import (
    BackoffPolicy,
    LiveVideoPipeline,
    MetadataPublisher,
    PipelineConfig,
    SourceFactory,
    StreamSource,
    opencv_source_factory,
)

from .api import RunnerCommsHandler, RunnerView
from .config import SystemConfig
from .detection import RunnerDetector
from .fusion_service import FusionService
from .plugins import PluginRegistry
from .simulators import (
    ActuatorSimulator,
    IdentitySimulator,
    LocationSimulator,
    TagReplay,
)
from .sources import PLACEHOLDER_URI, FileSourceFactory, fake_source_factory
from .stats import MemoryAlertSink, MemoryEventSink
from .tracking import ZoneEntryRule, make_track_fn

logger = logging.getLogger(__name__)

_DETECTION_KEYS = ("xyxy", "cls", "conf", "label")


class Component(Protocol):
    """Pieza opcional del sistema con ciclo de vida y salud en metadata."""

    name: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def health(self) -> dict[str, Any]: ...


class SystemApp:
    """Sistema completo: runtime + pipeline + fusión + plugins/simuladores."""

    def __init__(
        self,
        config: SystemConfig,
        *,
        hub: InMemoryHub | None = None,
        plugins: PluginRegistry | None = None,
        source_factory: SourceFactory | None = None,
        detector: Detector | None = None,
        api_token: str | None = None,
    ) -> None:
        self.config = config
        self._hub: InMemoryHub = hub or InMemoryHub(
            queue_size=config.queue_size, history_size=config.history_size
        )
        self._api_token = api_token or None
        self._tap = BusTap(self._hub)
        self._view = RunnerView(self._tap, self._component_health)
        self._server: CommsServer | None = None
        self._plugins = plugins or PluginRegistry()
        self._source_factory = source_factory
        self._detection = RunnerDetector(config.detector, detector)
        self._event_sink = MemoryEventSink(config.recent_limit)
        self._alert_sink = MemoryAlertSink(config.recent_limit)
        self._runtime: AgentRuntime | None = None
        self._pipeline: LiveVideoPipeline | None = None
        self._components: list[Component] = []
        self._by_name: dict[str, Component] = {}
        self._health_task: asyncio.Task[None] | None = None
        self._started_at: float | None = None
        self._running = False
        self._stopped = False

    # ------------------------------------------------------------------ API

    @property
    def hub(self) -> InMemoryHub:
        return self._hub

    @property
    def tap(self) -> BusTap:
        """Tap del bus: única fuente de estadísticas por tópico."""
        return self._tap

    @property
    def view(self) -> RunnerView:
        """``SystemView`` (agentes, tópicos, eventos, decisiones) de la API."""
        return self._view

    @property
    def api_url(self) -> str | None:
        """URL base de la API de observabilidad, o ``None`` si está apagada."""
        if self._server is None or not self._running:
            return None
        host = self.config.api.host
        shown = f"[{host}]" if ":" in host else host
        return f"http://{shown}:{self._server.port}"

    @property
    def runtime(self) -> AgentRuntime:
        if self._runtime is None:
            raise RuntimeError("El sistema aún no se ha iniciado")
        return self._runtime

    @property
    def plugins(self) -> PluginRegistry:
        return self._plugins

    async def start(self) -> None:
        if self._running:
            raise RuntimeError("El sistema ya está en ejecución")
        if self._stopped:
            raise RuntimeError("El sistema ya fue detenido; cree un SystemApp nuevo")
        cfg = self.config
        self._started_at = time.monotonic()
        try:
            self._build_runtime()
            assert self._runtime is not None
            await self._runtime.start()
            self._view.bind_runtime(self._runtime)
            self._running = True
            await self._start_components()
            await self._start_pipeline()
            self._health_task = asyncio.create_task(
                self._health_loop(), name="system:health"
            )
        except BaseException:
            await self._shutdown()
            raise
        logger.info("Sistema iniciado (perfil %s)", cfg.profile)

    async def stop(self) -> None:
        if not self._running:
            return
        await self._shutdown()

    async def run(
        self,
        *,
        duration: float | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Inicia, espera (duración, evento o cancelación) y apaga limpio."""
        await self.start()
        try:
            event = stop_event or asyncio.Event()
            if duration is None:
                await event.wait()
            else:
                with suppress(TimeoutError):
                    await asyncio.wait_for(event.wait(), duration)
        finally:
            await self.stop()

    def snapshot(self) -> dict[str, Any]:
        """Perfil, componentes y pipeline como JSON estricto.

        Agentes, tópicos, eventos y decisiones se leen de ``SystemApp.view``
        (tap del bus), no de aquí.
        """
        return {
            "profile": self.config.profile,
            "running": self._running,
            "uptime_s": round(time.monotonic() - self._started_at, 3)
            if self._started_at is not None and self._running
            else 0.0,
            "components": self._component_health(),
            "pipeline": dict(self._pipeline.health()) if self._pipeline else {},
            "api": self.api_url,
        }

    # ------------------------------------------------------------ ensamblado

    def _build_runtime(self) -> None:
        cfg = self.config
        handlers = build_default_handlers(
            storage_sink=self._event_sink,
            alert_sink=self._alert_sink,
            event_rules=[ZoneEntryRule(cfg.zones)],
            track_fn=make_track_fn(cfg.zones),
        )
        api = cfg.api
        if api.enabled:
            check_bind(api.host, self._api_token)  # falla antes de arrancar nada
            web = create_app(
                self._view,
                token=self._api_token,
                host=api.host,
                allowed_origins=api.allowed_origins,
            )
            self._server = CommsServer(web, api.host, api.port)
        handlers["comms"] = RunnerCommsHandler(
            self._alert_sink,
            self._tap,
            self._server,
            (api.host, api.port) if api.enabled else None,
        )
        self._runtime = AgentRuntime(
            self._hub,
            handlers,
            heartbeat_interval=cfg.heartbeat_interval_s,
            shutdown_timeout=cfg.shutdown_timeout_s,
        )

    def _register(self, component: Component) -> Component:
        self._components.append(component)
        self._by_name[component.name] = component
        return component

    async def _start_components(self) -> None:
        cfg = self.config
        self._register(FusionService(self._hub, cfg))
        tag = cfg.tag.kind
        if tag == "sim":
            self._register(LocationSimulator(self._hub, cfg))
        elif tag == "replay":
            self._register(TagReplay(self._hub, cfg))
        elif tag == "c6":
            self._register(self._c6_fallback())
        if cfg.identity.kind == "sim":
            self._register(IdentitySimulator(self._hub, cfg))
        if cfg.actuator.kind == "sim":
            self._register(ActuatorSimulator(self._hub, cfg))
        for component in self._components:
            await component.start()

    def _c6_fallback(self) -> LocationSimulator:
        """Sin C6 utilizable, la ubicación queda ``unknown`` y el sistema sigue."""
        if self._plugins.is_installed("tagbridge"):
            detail = (
                "tagbridge instalado, pero el adaptador del ejecutor aún no lo "
                "conecta: ubicación unknown"
            )
        else:
            detail = (
                "tagbridge no instalado (llega con #37): ubicación unknown hasta "
                "conectar el C6"
            )
        return LocationSimulator(
            self._hub, self.config, degraded_reason="tag_missing", detail=detail
        )

    # --------------------------------------------------------------- pipeline

    def _build_source_factory(self) -> tuple[SourceFactory, str]:
        cam = self.config.camera
        if self._source_factory is not None:
            return self._source_factory, PLACEHOLDER_URI
        if cam.kind == "fake":
            return fake_source_factory(cam.stream_id, cam.fps), PLACEHOLDER_URI
        if cam.kind == "file":
            return FileSourceFactory(cam.path, cam.fps), PLACEHOLDER_URI
        return opencv_source_factory, f"usb:{cam.device_index}"

    async def _start_pipeline(self) -> None:
        cfg = self.config
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._detection.prepare)
        factory, uri = self._build_source_factory()
        publisher = MetadataPublisher(self._hub, loop, source="pipeline")
        self._pipeline = LiveVideoPipeline(
            factory,
            config=PipelineConfig(
                backoff=BackoffPolicy(initial=1.0, maximum=10.0),
                join_timeout=cfg.shutdown_timeout_s,
            ),
            processor=self._detection.processor,
            publisher=publisher,
        )
        self._pipeline.add_source(StreamSource(cfg.camera.stream_id, uri))
        await loop.run_in_executor(None, self._pipeline.start)

    # ----------------------------------------------------------------- salud

    def _camera_health(self) -> dict[str, Any]:
        cam = self.config.camera
        base: dict[str, Any] = {"kind": cam.kind, "stream_id": cam.stream_id}
        if self._pipeline is None:
            return {**base, "status": "stopped", "detail": "pipeline detenido"}
        try:
            h = self._pipeline.stream_health(cam.stream_id).to_payload()
        except KeyError:
            return {**base, "status": "stopped", "detail": "sin stream"}
        state = h["state"]
        base.update(
            frames_received=h["frames_received"],
            frames_dropped=h["frames_dropped"],
            retry_count=h["retry_count"],
            last_error=h["last_error"],
            stream_state=state,
        )
        if state == "connected":
            return {**base, "status": "ok", "detail": "recibiendo frames"}
        if state == "stopped":
            return {**base, "status": "stopped", "detail": "pipeline detenido"}
        if state in ("idle", "connecting") and not h["last_error"]:
            return {**base, "status": "starting", "detail": state}
        return {
            **base,
            "status": "degraded",
            "detail": h["last_error"] or state,
        }

    def _component_health(self) -> dict[str, dict[str, Any]]:
        cfg = self.config
        out: dict[str, dict[str, Any]] = {
            "camera": self._camera_health(),
            "detector": self._detection.health(),
        }
        fusion = self._by_name.get("fusion")
        out["fusion"] = (
            fusion.health() if fusion else {"status": "stopped", "detail": ""}
        )
        provided = {
            "location": (self._by_name.get("location"), cfg.tag.kind),
            "identity": (self._by_name.get("identity"), cfg.identity.kind),
            "actuation": (self._by_name.get("actuation"), cfg.actuator.kind),
        }
        for name, (component, kind) in provided.items():
            plugin = self._plugins.health(name)
            if component is not None:
                entry = component.health()
            else:
                entry = {"status": "disabled", "detail": "desactivado en el perfil"}
            out[name] = {**entry, "kind": kind, "plugin": plugin["status"]}
        tag_plugin = self._plugins.health("tagbridge")
        if cfg.tag.kind == "c6":
            location = out["location"]
            tag_entry = {"status": "degraded", "detail": location.get("detail", "")}
        elif cfg.tag.kind == "none":
            tag_entry = {"status": "disabled", "detail": "desactivado en el perfil"}
        else:
            tag_entry = {
                "status": "not_used",
                "detail": f"el perfil usa tag {cfg.tag.kind}",
            }
        out["tagbridge"] = {
            **tag_entry,
            "kind": cfg.tag.kind,
            "plugin": tag_plugin["status"],
        }
        return out

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.heartbeat_interval_s)
            try:
                for name, health in self._component_health().items():
                    payload: Mapping[str, Any] = {
                        **health,
                        "kind": "component.health",
                        "component": name,
                    }
                    payload = {
                        k: v for k, v in payload.items() if k not in ("state", "role")
                    }
                    await self._hub.publish(
                        Topic.HEALTH,
                        MetadataEnvelope(source=f"component/{name}", payload=payload),
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("No se pudo publicar la salud de componentes")

    # ---------------------------------------------------------------- parada

    async def _shutdown(self) -> None:
        """Apaga en orden: salud, video, productores, runtime, hub, detector."""
        loop = asyncio.get_running_loop()
        self._stopped = True
        if self._health_task is not None:
            task, self._health_task = self._health_task, None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._pipeline is not None:
            pipeline = self._pipeline
            await loop.run_in_executor(None, pipeline.stop)
        for component in reversed(self._components):
            try:
                await component.stop()
            except Exception:
                logger.exception("Error al detener '%s'", component.name)
        if self._runtime is not None and self._running:
            with suppress(TimeoutError):
                await self._runtime.wait_idle(timeout=self.config.shutdown_timeout_s)
            await self._runtime.stop()
        await self._hub.close()
        await loop.run_in_executor(None, self._detection.close)
        self._running = False
