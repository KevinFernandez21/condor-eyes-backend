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
import secrets
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
from .cameras import CameraStatusPublisher, ScenarioDirector, camera_state
from .config import SystemConfig
from .detection import RunnerDetector
from .fusion_service import FusionService
from .plugins import PluginRegistry
from .pseudonym import Pseudonymizer
from .scenes import SceneDetector, SceneSimulator
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
        seed: int | None = None,
    ) -> None:
        self.config = config
        self._seed: int = secrets.randbits(32) if seed is None else seed
        self._ps = Pseudonymizer.from_seed(self._seed)
        self._ps.check_unique(config.permissions)
        self._scene: SceneSimulator | None = None
        if (
            detector is None
            and config.detector.kind == "scenes"
            and config.multi_camera
        ):
            self._scene = SceneSimulator(
                config.cameras,
                config.scenes,
                handles=tuple(config.permissions),
                seed=self._seed,
            )
            detector = SceneDetector(self._scene)
        self._fake_factory: Any = None
        self._factory: SourceFactory | None = None
        self._uri = PLACEHOLDER_URI
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
    def seed(self) -> int:
        """Semilla de la simulación (escenas, caídas y seudónimos)."""
        return self._seed

    @property
    def pseudonymizer(self) -> Pseudonymizer:
        return self._ps

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
            "seed": self._seed,
            "cameras": self._cameras_latest(),
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
            event_rules=[ZoneEntryRule(cfg.effective_zones())],
            track_fn=make_track_fn(
                cfg.effective_zones(),
                {c.camera_id: c.zones for c in cfg.cameras}
                if cfg.multi_camera
                else None,
            ),
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
        self._factory, self._uri = self._build_source_factory()
        self._register(FusionService(self._hub, cfg, self._ps))
        self._register(CameraStatusPublisher(self._hub, cfg, self._streams))
        if self._fake_factory is not None and cfg.multi_camera:
            self._register(
                ScenarioDirector(self._hub, cfg, self._fake_factory, seed=self._seed)
            )
        tag = cfg.tag.kind
        if tag == "sim":
            self._register(
                LocationSimulator(
                    self._hub, cfg, pseudonymizer=self._ps, scene=self._scene
                )
            )
        elif tag == "replay":
            self._register(TagReplay(self._hub, cfg, pseudonymizer=self._ps))
        elif tag == "c6":
            self._register(self._c6_fallback())
        if cfg.identity.kind == "sim":
            self._register(
                IdentitySimulator(
                    self._hub,
                    cfg,
                    pseudonymizer=self._ps,
                    scene=self._scene,
                    seed=self._seed,
                )
            )
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
            self._fake_factory = fake_source_factory(self.config.camera_specs())
            return self._fake_factory, PLACEHOLDER_URI
        if cam.kind == "file":
            return FileSourceFactory(cam.path, cam.fps), PLACEHOLDER_URI
        return opencv_source_factory, f"usb:{cam.device_index}"

    async def _start_pipeline(self) -> None:
        cfg = self.config
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._detection.prepare)
        assert self._factory is not None
        factory, uri = self._factory, self._uri
        publisher = MetadataPublisher(self._hub, loop, source="pipeline")
        self._pipeline = LiveVideoPipeline(
            factory,
            config=PipelineConfig(
                backoff=(
                    BackoffPolicy(initial=0.25, maximum=2.0)
                    if cfg.camera.kind == "fake"
                    else BackoffPolicy(initial=1.0, maximum=10.0)
                ),
                join_timeout=cfg.shutdown_timeout_s,
            ),
            processor=self._detection.processor,
            publisher=publisher,
        )
        for spec in cfg.camera_specs():
            self._pipeline.add_source(StreamSource(spec.camera_id, uri))
        await loop.run_in_executor(None, self._pipeline.start)

    # ----------------------------------------------------------------- salud

    def _streams(self) -> Mapping[str, Mapping[str, Any]]:
        """Salud por stream del pipeline (fuente de ``CameraStatusPublisher``)."""
        if self._pipeline is None:
            return {}
        streams = self._pipeline.health().get("streams", {})
        return streams if isinstance(streams, Mapping) else {}

    def _cameras_latest(self) -> list[dict[str, Any]]:
        publisher = self._by_name.get("cameras")
        return publisher.latest if isinstance(publisher, CameraStatusPublisher) else []

    def _camera_health(self) -> dict[str, Any]:
        """Estado agregado de las cámaras (el detalle por cámara va en ``cameras``)."""
        cfg = self.config
        specs = cfg.camera_specs()
        base: dict[str, Any] = {
            "kind": cfg.camera.kind,
            "stream_id": specs[0].camera_id,
            "total": len(specs),
        }
        if self._pipeline is None:
            return {**base, "status": "stopped", "detail": "pipeline detenido"}
        streams = self._streams()
        ok = starting = stopped = 0
        problems: list[str] = []
        received = dropped = 0
        stream_state = None
        for spec in specs:
            stream = streams.get(spec.camera_id)
            if stream is None:
                problems.append(f"{spec.name}: sin stream")
                continue
            state = camera_state(stream)
            received += int(stream.get("frames_received") or 0)
            dropped += int(stream.get("frames_dropped") or 0)
            stream_state = stream.get("state")
            if stream_state == "stopped":
                stopped += 1
            elif state[0] == "ok":
                ok += 1
            elif stream_state in ("idle", "connecting") and not stream.get(
                "last_error"
            ):
                starting += 1
            else:
                problems.append(f"{spec.name}: {state[0]} ({state[1]})")
        base.update(
            ok=ok,
            frames_received=received,
            frames_dropped=dropped,
            stream_state=stream_state,
            last_error=problems[0] if problems else None,
        )
        if stopped == len(specs):
            return {**base, "status": "stopped", "detail": "pipeline detenido"}
        if problems:
            return {**base, "status": "degraded", "detail": "; ".join(problems)}
        if starting:
            return {**base, "status": "starting", "detail": f"{ok}/{len(specs)} ok"}
        return {**base, "status": "ok", "detail": "recibiendo frames"}

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
        director = self._by_name.get("scenarios")
        if director is not None:
            out["scenarios"] = director.health()
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
