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
import math
import os
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from agents import AgentRuntime
from agents.handlers import build_default_handlers
from bus import InMemoryHub, MetadataEnvelope, Topic
from compare import FakeDetector
from compare.detectors import Detector
from pipeline import (
    BackoffPolicy,
    Frame,
    LiveVideoPipeline,
    MetadataPublisher,
    PipelineConfig,
    SourceFactory,
    StreamSource,
    opencv_source_factory,
)

from .config import SystemConfig
from .fusion_service import FusionService
from .plugins import PluginRegistry
from .simulators import (
    ActuatorSimulator,
    IdentitySimulator,
    LocationSimulator,
    MovingDetector,
    TagReplay,
)
from .sources import PLACEHOLDER_URI, FileSourceFactory, fake_source_factory
from .stats import InstrumentedHub, MemoryAlertSink, MemoryEventSink
from .tracking import ZoneEntryRule, make_track_fn

logger = logging.getLogger(__name__)

_DETECTION_KEYS = ("xyxy", "cls", "conf", "label")


class Component(Protocol):
    """Pieza opcional del sistema con ciclo de vida y salud en metadata."""

    name: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def health(self) -> dict[str, Any]: ...


def _clean_detections(raw: Any) -> list[dict[str, Any]]:
    """Reduce la salida del detector a primitivas JSON finitas (sin numpy)."""
    out: list[dict[str, Any]] = []
    for det in raw or []:
        try:
            box = [float(v) for v in det["xyxy"]]
            item: dict[str, Any] = {
                "xyxy": box,
                "cls": int(det["cls"]),
                "conf": float(det["conf"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if len(box) != 4 or not all(math.isfinite(v) for v in (*box, item["conf"])):
            continue
        if isinstance(det.get("label"), str):
            item["label"] = det["label"]
        out.append(item)
    return out


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
    ) -> None:
        self.config = config
        self._hub: InMemoryHub = hub or InstrumentedHub(
            queue_size=config.queue_size, history_size=config.history_size
        )
        self._plugins = plugins or PluginRegistry()
        self._source_factory = source_factory
        self._detector: Detector | None = detector
        self._detector_injected = detector is not None
        self._detector_lock = threading.Lock()  # infer en curso vs close
        self._closing = False
        self._detector_state: dict[str, Any] = {
            "status": "starting",
            "detail": "",
            "name": None,
            "errors": 0,
        }
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
        """Estado actual como JSON estricto (solo metadata operativa)."""
        agents: dict[str, Any] = {}
        if self._runtime is not None:
            for name, health in self._runtime.health().items():
                agents[name] = {
                    **health,
                    "state": str(getattr(health["state"], "value", health["state"])),
                }
        stats = getattr(self._hub, "topic_stats", None)
        topics = (
            stats()
            if callable(stats)
            else {t.value: {"published": 0, "last_at": None} for t in Topic}
        )
        return {
            "profile": self.config.profile,
            "running": self._running,
            "uptime_s": round(time.monotonic() - self._started_at, 3)
            if self._started_at is not None and self._running
            else 0.0,
            "agents": agents,
            "topics": topics,
            "hub": dict(self._hub.stats),
            "components": self._component_health(),
            "pipeline": dict(self._pipeline.health()) if self._pipeline else {},
            "recent": {
                "events_stored": self._event_sink.total,
                "alerts_sent": self._alert_sink.total,
            },
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

    def _make_detector(self) -> Detector | None:
        kind = self.config.detector.kind
        if kind == "none":
            return None
        if kind == "fake":
            return FakeDetector(latency_ms=1.0)
        if kind == "moving":
            return MovingDetector()
        return self._build_yolo()

    def _weights_path(self) -> str:
        """Prioridad: ``CONDOR_WEIGHTS`` > ``[detector].weights`` > surveillance.toml."""
        det = self.config.detector
        env = os.environ.get("CONDOR_WEIGHTS", "").strip()
        if env:
            return env
        if det.weights:
            return det.weights
        from surveillance.detector import load_surveillance_config

        weights = load_surveillance_config(det.config_path).model
        if not Path(weights).is_absolute():
            weights = str(Path(det.config_path).resolve().parent.parent / weights)
        return weights

    def _build_yolo(self) -> Detector:
        det = self.config.detector
        weights = self._weights_path()
        if not Path(weights).is_file():
            raise FileNotFoundError(
                f"Pesos de YOLOv8n no encontrados: {weights}. Colóquelos en esa ruta, "
                "o indique otra con la variable de entorno CONDOR_WEIGHTS o con "
                "[detector].weights en configs/system.toml. Se obtienen del asset "
                "oficial 'yolov8n.pt' de ultralytics; no se descarga nada automáticamente."
            )
        from surveillance.detector import SurveillanceDetector, load_surveillance_config

        device = "0"
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"
        return SurveillanceDetector(
            load_surveillance_config(det.config_path, model=weights, device=device)
        )

    def _prepare_detector(self) -> None:
        """Construye y calienta el detector (bloqueante; corre en un hilo)."""
        state = self._detector_state
        try:
            if not self._detector_injected:
                self._detector = self._make_detector()
            if self._detector is None:
                state.update(status="disabled", detail="sin detector", name=None)
                return
            state["name"] = self._detector.name
            self._detector.warmup(1)
            state.update(status="ok", detail=self._detector.name)
        except Exception as exc:  # noqa: BLE001 - sin detector el sistema sigue (sin detecciones)
            state.update(
                status="degraded",
                detail=f"{type(exc).__name__}: {exc}",
                errors=state["errors"] + 1,
            )
            self._close_detector()
            logger.warning("Detector degradado: %s", state["detail"])

    def _close_detector(self) -> None:
        """Cierra el detector esperando a que termine la inferencia en curso."""
        self._closing = True
        with self._detector_lock:
            detector, self._detector = self._detector, None
        if detector is not None:
            with suppress(Exception):
                detector.close()

    def _process(self, stream_id: str, frame: Frame) -> list[Mapping[str, Any]] | None:
        """Inferencia del lado del pipeline: del frame solo salen detecciones."""
        if self._closing:
            return None
        with self._detector_lock:
            detector = self._detector
            if detector is None or self._closing:
                return None
            try:
                raw = detector.infer(frame.data)
            except Exception as exc:  # noqa: BLE001 - un frame malo no tumba el pipeline
                state = self._detector_state
                state.update(
                    status="degraded",
                    detail=f"{type(exc).__name__}: {exc}",
                    errors=state["errors"] + 1,
                )
                return None
        try:
            detections: list[Mapping[str, Any]] = list(_clean_detections(raw))
        except Exception as exc:  # noqa: BLE001 - un frame malo no tumba el pipeline
            state = self._detector_state
            state.update(
                status="degraded",
                detail=f"{type(exc).__name__}: {exc}",
                errors=state["errors"] + 1,
            )
            return None
        if self._detector_state["status"] == "degraded":
            self._detector_state.update(status="ok", detail=detector.name)
        return detections

    async def _start_pipeline(self) -> None:
        cfg = self.config
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._prepare_detector)
        factory, uri = self._build_source_factory()
        publisher = MetadataPublisher(self._hub, loop, source="pipeline")
        self._pipeline = LiveVideoPipeline(
            factory,
            config=PipelineConfig(
                backoff=BackoffPolicy(initial=1.0, maximum=10.0),
                join_timeout=cfg.shutdown_timeout_s,
            ),
            processor=self._process,
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
            "detector": dict(self._detector_state),
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
        await loop.run_in_executor(None, self._close_detector)
        self._running = False
