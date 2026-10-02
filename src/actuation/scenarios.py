"""Escenarios de lazo cerrado sobre el nodo SIMULADO (no es hardware).

Une controlador, adaptador y simulador en un solo objeto para probar
convergencia, oscilación, timeouts y paradas de emergencia con tiempo virtual.
Todo lo medido aquí es simulación y debe etiquetarse como tal.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .adapter import PanTiltActuator
from .config import ActuationConfig
from .controller import TargetObservation
from .simulator import ManualClock, SimulatedPanTiltNode, SimulatedTransport

TargetFn = Callable[[float], tuple[float, float] | None]


@dataclass(frozen=True, slots=True)
class TraceSample:
    """Estado real (sin ruido de medición) de la cámara en un instante simulado."""

    t: float
    pan: float
    tilt: float
    error_x: float  # error real normalizado (0.0 si no hay objetivo)
    error_y: float
    target_visible: bool


class _GatedTransport:
    """Envuelve un transporte para simular un corte total del enlace."""

    def __init__(self, inner: SimulatedTransport) -> None:
        self.inner = inner
        self.up = True
        self.kind = inner.kind

    def send(self, frame: bytes) -> None:
        if self.up:
            self.inner.send(frame)

    def recv(self) -> list[bytes]:
        frames = self.inner.recv()
        return frames if self.up else []

    def close(self) -> None:
        self.inner.close()


class ClosedLoopSim:
    """Cámara Pan-Tilt simulada que sigue un objetivo del mundo con visión retrasada.

    ``target_fn(t)`` devuelve el ángulo (pan, tilt) del objetivo en el mundo o
    ``None`` si no hay objetivo visible. La visión observa la pose de la cámara
    de hace ``vision_latency_s`` segundos, con ruido gaussiano ``obs_noise``.
    """

    def __init__(
        self,
        cfg: ActuationConfig,
        target_fn: TargetFn,
        vision_latency_s: float = 0.1,
        obs_noise: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.physics_dt = 0.005
        self._target_fn = target_fn
        self._vision_latency = vision_latency_s
        self._obs_noise = obs_noise
        self._rng = random.Random(cfg.simulator.seed if seed is None else seed)
        self.clock = ManualClock()
        self.node = SimulatedPanTiltNode(cfg, self.clock)
        self._link = SimulatedTransport(self.node, self.clock, cfg.simulator)
        self._gate = _GatedTransport(self._link)
        self.actuator = PanTiltActuator(self._gate, cfg, self.clock)
        self.actuator.start()
        self.trace: list[TraceSample] = []
        self._poses: deque[tuple[float, float, float]] = deque()
        self._next_tick = 0.0

    @property
    def link_up(self) -> bool:
        return self._gate.up

    @link_up.setter
    def link_up(self, value: bool) -> None:
        self._gate.up = value

    def restart_host(self) -> None:
        """Simula un reinicio del host: adaptador nuevo (seq a 0) sobre el mismo nodo."""
        self.actuator = PanTiltActuator(self._gate, self.cfg, self.clock)
        self.actuator.start()

    def inject(self, frame: bytes) -> None:
        """Inyecta una trama directamente al nodo (p. ej. para simular un replay)."""
        self._link.send(frame)

    def run(self, seconds: float) -> list[TraceSample]:
        """Avanza ``seconds`` de tiempo virtual; devuelve las muestras de ese tramo."""
        start = len(self.trace)
        period = 1.0 / self.cfg.control.control_rate_hz
        steps = round(seconds / self.physics_dt)
        for _ in range(steps):
            self.clock.advance(self.physics_dt)
            self._link.pump()
            now = self.clock.now()
            pan, tilt = self.node.position
            self._poses.append((now, pan, tilt))
            if now >= self._next_tick - 1e-12:
                self._next_tick += period
                self._control_tick(now)
            self.trace.append(self._sample(now, pan, tilt))
        return self.trace[start:]

    # --- internos ---

    def _pose_at(self, t: float) -> tuple[float, float]:
        poses = self._poses
        while len(poses) > 2 and poses[1][0] <= t:
            poses.popleft()
        return poses[0][1], poses[0][2]

    def _errors(
        self, target: tuple[float, float], pose: tuple[float, float]
    ) -> tuple[float, float]:
        ctl = self.cfg.control
        ex = (target[0] - pose[0]) / (ctl.hfov_deg / 2)
        ey = -(target[1] - pose[1]) / (
            ctl.vfov_deg / 2
        )  # y de imagen crece hacia abajo
        return min(max(ex, -1.0), 1.0), min(max(ey, -1.0), 1.0)

    def _control_tick(self, now: float) -> None:
        captured = now - self._vision_latency
        target = self._target_fn(captured)
        observation = None
        if target is not None:
            ex, ey = self._errors(target, self._pose_at(captured))
            if self._obs_noise:
                ex += self._rng.gauss(0.0, self._obs_noise)
                ey += self._rng.gauss(0.0, self._obs_noise)
            observation = TargetObservation(ex, ey, captured, track_id=1)
        self.actuator.track(observation)

    def _sample(self, now: float, pan: float, tilt: float) -> TraceSample:
        target = self._target_fn(now)
        if target is None:
            return TraceSample(now, pan, tilt, 0.0, 0.0, False)
        ex, ey = self._errors(target, (pan, tilt))
        return TraceSample(now, pan, tilt, ex, ey, True)
