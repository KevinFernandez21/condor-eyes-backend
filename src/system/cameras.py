"""Metadata por cámara y caídas simuladas para la maqueta multicámara.

``CameraStatusPublisher`` publica en ``Topic.STREAM_STATUS`` un mensaje
``kind: "camera.status"`` por cámara (nombre visible, escena, zonas, fps,
resolución y estado ``ok`` / ``degraded`` / ``offline``). ``ScenarioDirector``
provoca caídas y recuperaciones aleatorias de las cámaras sintéticas.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Mapping
from typing import Any

from bus import MetadataHub, Topic
from pipeline.fake import FakeSourceFactory

from .config import SystemConfig
from .simulators import PeriodicComponent

CAMERA_KIND = "camera.status"
OFFLINE_AFTER_RETRIES = 3

StreamsSource = Callable[[], Mapping[str, Mapping[str, Any]]]


def camera_state(stream: Mapping[str, Any] | None) -> tuple[str, str]:
    """Estado de cámara (``ok``/``degraded``/``offline``) y detalle desde el pipeline."""
    if stream is None:
        return "offline", "sin stream"
    state = str(stream.get("state"))
    error = stream.get("last_error")
    retries = int(stream.get("retry_count") or 0)
    if state == "connected":
        return "ok", "recibiendo frames"
    if state in ("failed", "stopped"):
        return "offline", str(error or state)
    if retries >= OFFLINE_AFTER_RETRIES:
        return "offline", str(error or state)
    return "degraded", str(error or state)


class CameraStatusPublisher(PeriodicComponent):
    """Estado de cada cámara por ``Topic.STREAM_STATUS`` (metadata por cámara)."""

    name = "cameras"

    def __init__(
        self, hub: MetadataHub, cfg: SystemConfig, streams: StreamsSource
    ) -> None:
        super().__init__(
            hub,
            min(1.0, cfg.heartbeat_interval_s),
            status="ok",
            detail=f"{len(cfg.camera_specs())} cámaras",
        )
        self._cfg = cfg
        self._streams = streams
        self._last: dict[str, tuple[int, float]] = {}
        self._latest: list[dict[str, Any]] = []

    def payloads(self) -> list[dict[str, Any]]:
        """Un payload por cámara con el estado actual (también para ``snapshot``)."""
        loop_time = asyncio.get_running_loop().time()
        streams = self._streams()
        out: list[dict[str, Any]] = []
        for spec in self._cfg.camera_specs():
            stream = streams.get(spec.camera_id)
            state, _ = camera_state(stream)
            frames = int((stream or {}).get("frames_received") or 0)
            previous = self._last.get(spec.camera_id)
            measured = 0.0
            if previous is not None and loop_time > previous[1]:
                measured = max(0.0, (frames - previous[0]) / (loop_time - previous[1]))
            self._last[spec.camera_id] = (frames, loop_time)
            out.append(
                {
                    "kind": CAMERA_KIND,
                    "camera_id": spec.camera_id,
                    "stream_id": spec.camera_id,
                    "name": spec.name,
                    "scene": spec.scene,
                    "zones": [
                        {"zone_id": z.zone_id, "restricted": z.restricted}
                        for z in spec.zones
                    ],
                    "fps": spec.fps,
                    "measured_fps": round(measured, 2),
                    "state": state,
                    "stream_state": (stream or {}).get("state"),
                    "resolution": {
                        "width": spec.width or None,
                        "height": spec.height or None,
                    },
                    "frames_received": frames,
                    "last_frame_at": (stream or {}).get("last_frame_at"),
                    "last_error": (stream or {}).get("last_error"),
                    "simulated": self._cfg.camera.kind == "fake",
                }
            )
        self._latest = out
        return out

    @property
    def latest(self) -> list[dict[str, Any]]:
        return list(self._latest)

    async def _tick(self) -> None:
        for payload in self.payloads():
            await self._publish(
                Topic.STREAM_STATUS, "cameras", payload, stream_id=payload["camera_id"]
            )


class ScenarioDirector(PeriodicComponent):
    """Caídas aleatorias de cámaras sintéticas (con recuperación), sembradas."""

    name = "scenarios"

    def __init__(
        self,
        hub: MetadataHub,
        cfg: SystemConfig,
        factory: FakeSourceFactory,
        *,
        seed: int,
    ) -> None:
        super().__init__(hub, 0.1, status="ok" if cfg.scenes.dropouts else "disabled")
        self._cfg = cfg
        self._factory = factory
        self._rng = random.Random(f"{seed}:director")
        self._ids = [c.camera_id for c in cfg.cameras]
        self._down: tuple[str, float] | None = (
            None  # (cámara, instante de recuperación)
        )
        self._next_at: float | None = None
        self.dropouts = 0

    def _schedule(self, now: float) -> None:
        mean = self._cfg.scenes.dropout_interval_s
        self._next_at = now + mean * self._rng.uniform(0.5, 1.5)

    async def _on_start(self) -> None:
        self._detail = "caídas activadas" if self._cfg.scenes.dropouts else "sin caídas"

    async def _on_stop(self) -> None:
        for camera in self._factory.cameras():
            camera.reconnect()  # nunca dejar una cámara caída al apagar
        self._down = None

    async def _tick(self) -> None:
        if not self._cfg.scenes.dropouts or not self._ids:
            return
        now = asyncio.get_running_loop().time()
        if self._next_at is None:
            self._schedule(now)
            return
        if self._down is not None:
            camera_id, until = self._down
            if now >= until:
                self._factory.camera(camera_id).reconnect()
                self._down = None
                self._detail = f"{camera_id} recuperada"
                self._schedule(now)
            return
        if now >= self._next_at:
            camera_id = self._rng.choice(self._ids)
            self._factory.camera(camera_id).disconnect()
            duration = self._cfg.scenes.dropout_duration_s * self._rng.uniform(0.7, 1.3)
            self._down = (camera_id, now + duration)
            self.dropouts += 1
            self._detail = f"{camera_id} caída"

    def health(self) -> dict[str, Any]:
        return {**super().health(), "dropouts": self.dropouts}
