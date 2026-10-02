"""Escenas sintéticas multicámara para la maqueta (perfil ``sim``).

Cada cámara tiene su propio generador aleatorio (``Random`` sembrado con
``seed`` y el id de la cámara) y avanza **por frame**, no por reloj de pared: la
secuencia de cada cámara es reproducible con ``--seed`` aunque los hilos se
intercalen distinto. Las personas se reparten por cámara los identificadores de
la config (``permissions``) para que ese reparto tampoco dependa del orden.

Una detección es solo metadata JSON estricta::

    {"stream_id", "track_id", "cls", "label", "conf", "xyxy", "frame_ts"}

``cls`` usa los índices de ``surveillance.classes.NAMES`` (0 persona, 2 coche,
4 bus, 5 camión); ``cls = -1`` / ``label = "motion"`` es movimiento sin clase.
"""

from __future__ import annotations

import random
import secrets
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from .config import CameraSpec, SceneConfig, ZoneConfig
from .tracking import zone_for

MOTION_CLS = -1
_VEHICLES = ((2, "car", 0.7), (5, "truck", 0.2), (4, "bus", 0.1))


class Role(StrEnum):
    """Perfil de una persona simulada (verdad de terreno para tag e identidad)."""

    STAFF = "staff"  # conocida y con tag: corrobora
    STAFF_NO_TAG = "staff_no_tag"  # conocida, sin tag: "persona sin tag"
    STRANGER = "stranger"  # desconocida y sin tag: intrusión si es zona restringida


@dataclass(frozen=True, slots=True)
class ActorProfile:
    """Verdad de terreno de un actor visible (o recién desaparecido)."""

    label: str
    role: Role | None  # solo personas
    handle: str | None
    zone_id: str | None


@dataclass(slots=True)
class _Actor:
    track_id: int
    cls: int
    label: str
    cx: float
    bottom: float
    w: float
    h: float
    vx: float
    conf: float
    role: Role | None = None
    handle: str | None = None
    dwell_x: float | None = None
    dwell_left: float = 0.0
    entered: bool = False
    zone_id: str | None = None


class _CameraScene:
    def __init__(
        self,
        spec: CameraSpec,
        cfg: SceneConfig,
        handles: Sequence[str],
        seed: int,
    ) -> None:
        self.spec = spec
        self._cfg = cfg
        self._handles = tuple(handles)
        self._rng = random.Random(f"{seed}:{spec.camera_id}")
        self._actors: list[_Actor] = []
        self._next_id = 0
        self._dt = 1.0 / spec.fps
        self.dead: OrderedDict[int, ActorProfile] = OrderedDict()

    @property
    def actors(self) -> list[_Actor]:
        return self._actors

    # -- altas --

    def _free_handle(self) -> str | None:
        held = {a.handle for a in self._actors if a.handle}
        free = [h for h in self._handles if h not in held]
        return self._rng.choice(free) if free else None

    def _spawn(self) -> None:
        rng, spec, cfg = self._rng, self.spec, self._cfg
        self._next_id += 1
        kind = rng.random()
        bottom = rng.uniform(0.55, 0.95) * spec.height
        depth = 0.6 + 0.8 * (bottom - 0.55 * spec.height) / (0.4 * spec.height)
        role: Role | None = None
        handle: str | None = None
        dwell_x: float | None = None
        if kind < cfg.motion_ratio:
            cls, label = MOTION_CLS, "motion"
            w = rng.uniform(50, 180)
            h = w * rng.uniform(0.6, 1.4)
            bottom = rng.uniform(0.3, 0.9) * spec.height
            speed = rng.uniform(120, 300)
            conf = rng.uniform(0.4, 0.8)
        elif kind < cfg.motion_ratio + cfg.vehicle_ratio:
            roll, acc = rng.random(), 0.0
            cls, label = _VEHICLES[0][0], _VEHICLES[0][1]
            for v_cls, v_label, weight in _VEHICLES:
                acc += weight
                if roll < acc:
                    cls, label = v_cls, v_label
                    break
            scale = {"car": 1.0, "truck": 1.4, "bus": 1.8}[label]
            w = 300 * depth * scale * rng.uniform(0.85, 1.2)
            h = w * (0.5 if label == "car" else 0.6)
            speed = rng.uniform(350, 700) * depth
            conf = rng.uniform(0.7, 0.98)
        else:
            cls, label = 0, "person"
            w = 80 * depth * rng.uniform(0.9, 1.1)
            h = w * rng.uniform(2.3, 2.8)
            speed = rng.uniform(90, 220) * depth
            conf = rng.uniform(0.55, 0.97)
            role, handle = self._assign_role()
            if rng.random() < 0.4:
                dwell_x = rng.uniform(0.15, 0.85) * spec.width
        direction = rng.choice((-1.0, 1.0))
        cx = -w / 2 if direction > 0 else spec.width + w / 2
        if dwell_x is not None and (dwell_x - cx) * direction < 0:
            dwell_x = None
        self._actors.append(
            _Actor(
                track_id=self._next_id,
                cls=cls,
                label=label,
                cx=cx,
                bottom=bottom,
                w=w,
                h=h,
                vx=speed * direction,
                conf=conf,
                role=role,
                handle=handle,
                dwell_x=dwell_x,
            )
        )

    def _assign_role(self) -> tuple[Role, str | None]:
        cfg, roll = self._cfg, self._rng.random()
        handle = self._free_handle()
        if handle is None or roll < cfg.stranger_ratio:
            return Role.STRANGER, None
        if roll < cfg.stranger_ratio + cfg.no_tag_ratio:
            return Role.STAFF_NO_TAG, handle
        return Role.STAFF, handle

    # -- avance --

    def _advance(self, actor: _Actor) -> bool:
        """Mueve un actor; False si ya salió del encuadre."""
        spec = self.spec
        if actor.dwell_left > 0:
            actor.dwell_left -= self._dt
        else:
            moved = actor.cx + actor.vx * self._dt
            if actor.dwell_x is not None and (
                (actor.cx - actor.dwell_x) * (moved - actor.dwell_x) <= 0
            ):
                actor.cx = actor.dwell_x
                actor.dwell_left = self._rng.uniform(3.0, 8.0)
                actor.dwell_x = None
            else:
                actor.cx = moved
        inside = -actor.w / 2 < actor.cx < spec.width + actor.w / 2
        if inside:
            actor.entered = True
        return inside or not actor.entered

    def step(self) -> list[dict[str, Any]]:
        rng, spec, cfg = self._rng, self.spec, self._cfg
        if len(self._actors) < cfg.max_actors and rng.random() < (
            cfg.spawn_rate_per_s * self._dt
        ):
            self._spawn()
        alive: list[_Actor] = []
        for actor in self._actors:
            if self._advance(actor):
                alive.append(actor)
            else:
                self._remember(actor)
        self._actors = alive
        now = datetime.now(UTC).isoformat()
        out: list[dict[str, Any]] = []
        for actor in alive:
            fraction = min(max(actor.cx / spec.width, 0.0), 0.999999)
            zone = zone_for(spec.zones, fraction)
            actor.zone_id = zone.zone_id if zone else None
            x0 = max(0.0, actor.cx - actor.w / 2 + rng.uniform(-1.5, 1.5))
            x1 = min(float(spec.width), actor.cx + actor.w / 2 + rng.uniform(-1.5, 1.5))
            y1 = min(float(spec.height), actor.bottom + rng.uniform(-1.0, 1.0))
            y0 = max(0.0, y1 - actor.h)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue  # casi fuera del encuadre
            conf = min(0.99, max(0.05, actor.conf + rng.uniform(-0.03, 0.03)))
            out.append(
                {
                    "stream_id": spec.camera_id,
                    "track_id": actor.track_id,
                    "cls": actor.cls,
                    "label": actor.label,
                    "conf": round(conf, 3),
                    "xyxy": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                    "frame_ts": now,
                }
            )
        return out

    def _remember(self, actor: _Actor) -> None:
        self.dead[actor.track_id] = self.profile_of(actor)
        while len(self.dead) > 200:
            self.dead.popitem(last=False)

    @staticmethod
    def profile_of(actor: _Actor) -> ActorProfile:
        return ActorProfile(actor.label, actor.role, actor.handle, actor.zone_id)


class SceneSimulator:
    """Todas las cámaras de la maqueta; ``step(camera_id)`` produce un frame."""

    def __init__(
        self,
        cameras: Sequence[CameraSpec],
        cfg: SceneConfig,
        *,
        handles: Sequence[str],
        seed: int | None = None,
    ) -> None:
        if not cameras:
            raise ValueError("Se necesita al menos una cámara")
        self.seed: int = secrets.randbits(32) if seed is None else seed
        self.cameras = tuple(cameras)
        self.zones: tuple[ZoneConfig, ...] = tuple(
            z for c in self.cameras for z in c.zones
        )
        count = len(self.cameras)
        self._scenes = {
            cam.camera_id: _CameraScene(cam, cfg, list(handles)[i::count], self.seed)
            for i, cam in enumerate(self.cameras)
        }
        self._lock = threading.Lock()

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self._scenes)

    def step(self, camera_id: str) -> list[dict[str, Any]]:
        scene = self._scenes[camera_id]
        with self._lock:
            return scene.step()

    def profile(self, track_ref: str) -> ActorProfile | None:
        """Verdad de terreno de ``<cámara>/<track_id>`` (vivo o recién terminado)."""
        camera_id, _, tail = track_ref.rpartition("/")
        scene = self._scenes.get(camera_id)
        if scene is None or not tail.isdigit():
            return None
        track_id = int(tail)
        with self._lock:
            for actor in scene.actors:
                if actor.track_id == track_id:
                    return scene.profile_of(actor)
            return scene.dead.get(track_id)

    def zone_of(self, handle: str) -> str | None:
        """Zona del actor vivo que tiene ``handle``, o ``None`` si no hay ninguno."""
        with self._lock:
            for scene in self._scenes.values():
                for actor in scene.actors:
                    if actor.handle == handle:
                        return actor.zone_id
        return None

    def holders(self) -> dict[str, Role]:
        """Rol de cada handle ahora mismo en escena."""
        with self._lock:
            return {
                a.handle: a.role
                for scene in self._scenes.values()
                for a in scene.actors
                if a.handle and a.role
            }


class SceneDetector:
    """``Detector`` sintético que sirve las escenas (``infer_stream`` usa el stream)."""

    name = "scenes-sim"

    def __init__(self, scenes: SceneSimulator) -> None:
        self._scenes = scenes

    def warmup(self, n: int = 5) -> None:
        return None

    def infer_stream(self, stream_id: str, frame: Any) -> list[dict]:
        return self._scenes.step(stream_id)

    def infer(self, frame: Any) -> list[dict]:
        return self._scenes.step(self._scenes.camera_ids[0])

    def close(self) -> None:
        return None
