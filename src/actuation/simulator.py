"""Simulador determinista del nodo ESP32-S3 Pan-Tilt (SIMULADO, no es hardware).

Modela, con tiempo virtual y semilla fija: latencia de enlace con jitter y
pérdida, tiempo muerto del servo, velocidad física limitada, ruido de medición
y la máquina de estados del firmware (secuencias, ESTOP enclavado, watchdog).

Todas las métricas obtenidas con este módulo son simuladas y deben etiquetarse
como tales; no sustituyen las mediciones de banco.
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, field

from .config import ActuationConfig, SimulatorConfig
from .protocol import (
    MS_MODULUS,
    Ack,
    AckStatus,
    ClearEstop,
    EmergencyStop,
    Heartbeat,
    Move,
    NodeState,
    ProtocolError,
    decode_command,
    encode_ack,
    seq_is_newer,
)


class ManualClock:
    """Reloj virtual: el tiempo solo avanza cuando el llamador lo indica."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("el reloj no puede retroceder")
        self._now += dt


@dataclass(order=True)
class _Pending:
    apply_at: float
    order: int
    pan: float = field(compare=False)
    tilt: float = field(compare=False)
    speed: float = field(compare=False)


class SimulatedPanTiltNode:
    """Réplica del comportamiento que debe tener el firmware del ESP32-S3."""

    def __init__(self, cfg: ActuationConfig, clock: ManualClock) -> None:
        self._lim = cfg.limits
        self._comms = cfg.comms
        self._sim = cfg.simulator
        self._clock = clock
        self._rng = random.Random(self._sim.seed)
        self._t = clock.now()
        self._pan = self._lim.neutral_pan_deg
        self._tilt = self._lim.neutral_tilt_deg
        self._target = (self._pan, self._tilt)
        self._speed = 0.0
        self._pending: list[_Pending] = []
        self._inbox: list[tuple[float, int, bytes]] = []
        self._order = itertools.count()
        self._last_seq: int | None = None
        self._last_rx: float | None = None
        self._estop = False
        self._failsafe = False
        self.crc_errors = 0
        self._boot_t = self._t

    # --- estado observable ---

    @property
    def position(self) -> tuple[float, float]:
        """Posición real (sin ruido) en grados respecto al neutro."""
        return (self._pan, self._tilt)

    @property
    def state(self) -> NodeState:
        if self._estop:
            return NodeState.ESTOP
        if self._failsafe:
            return NodeState.FAILSAFE
        if self._pending or (self._pan, self._tilt) != self._target:
            return NodeState.MOVING
        return NodeState.IDLE

    def reboot(self) -> None:
        """Reinicio del nodo (p. ej. brownout): neutro, sin ESTOP, secuencias y ms a cero."""
        self._physics_to(self._clock.now())
        self._inbox.clear()
        self._pending.clear()
        self._pan = self._lim.neutral_pan_deg
        self._tilt = self._lim.neutral_tilt_deg
        self._target = (self._pan, self._tilt)
        self._speed = 0.0
        self._last_seq = None
        self._last_rx = None
        self._estop = False
        self._failsafe = False
        self._boot_t = self._t

    # --- entrada / avance ---

    def deliver(self, frame: bytes, arrival: float) -> None:
        """Encola una trama que llegará al nodo en el instante ``arrival``."""
        heapq.heappush(self._inbox, (arrival, next(self._order), frame))

    def advance_to(self, until: float) -> list[tuple[float, bytes]]:
        """Avanza física y procesa tramas hasta ``until``; devuelve (t, ack) emitidos."""
        out: list[tuple[float, bytes]] = []
        while self._inbox and self._inbox[0][0] <= until:
            arrival, _, frame = heapq.heappop(self._inbox)
            self._physics_to(max(arrival, self._t))
            reply = self._handle(frame)
            if reply is not None:
                out.append((self._t, reply))
        self._physics_to(until)
        return out

    # --- física ---

    def _physics_to(self, until: float) -> None:
        dt_max = self._sim.physics_dt_s
        while self._t < until - 1e-12:
            step = min(dt_max, until - self._t)
            self._t += step
            while self._pending and self._pending[0].apply_at <= self._t:
                p = heapq.heappop(self._pending)
                self._target = (p.pan, p.tilt)
                self._speed = p.speed
            self._check_watchdog()
            reach = self._speed * step
            self._pan += max(-reach, min(reach, self._target[0] - self._pan))
            self._tilt += max(-reach, min(reach, self._target[1] - self._tilt))

    def _freeze(self) -> None:
        self._pending.clear()
        self._target = (self._pan, self._tilt)
        self._speed = 0.0

    def _check_watchdog(self) -> None:
        if (
            self._last_rx is not None
            and not self._failsafe
            and self._t - self._last_rx > self._comms.node_watchdog_s
        ):
            self._failsafe = True
            self._freeze()

    # --- protocolo ---

    def _ack(self, seq: int, status: AckStatus) -> bytes:
        noise = self._sim.noise_std_deg
        pan = self._pan + (self._rng.gauss(0.0, noise) if noise else 0.0)
        tilt = self._tilt + (self._rng.gauss(0.0, noise) if noise else 0.0)
        node_ms = int((self._t - self._boot_t) * 1000) % MS_MODULUS
        return encode_ack(
            Ack(seq, status, self.state, pan, tilt, node_ms, last_seq=self._last_seq)
        )

    def _handle(self, frame: bytes) -> bytes | None:
        try:
            cmd = decode_command(frame)
        except ProtocolError:
            self.crc_errors += 1
            return None
        self._last_rx = self._t
        newer = self._last_seq is None or seq_is_newer(cmd.seq, self._last_seq)

        if isinstance(
            cmd, EmergencyStop
        ):  # se atiende siempre, aunque el seq sea viejo
            if newer:
                self._last_seq = cmd.seq
            self._estop = True
            self._freeze()
            return self._ack(cmd.seq, AckStatus.OK)

        if not newer:
            status = (
                AckStatus.DUPLICATE if cmd.seq == self._last_seq else AckStatus.STALE
            )
            return self._ack(cmd.seq, status)
        self._last_seq = cmd.seq

        if isinstance(cmd, ClearEstop):
            self._estop = False
            return self._ack(cmd.seq, AckStatus.OK)
        if self._estop:
            return self._ack(cmd.seq, AckStatus.ESTOP)
        self._failsafe = False
        if isinstance(cmd, Heartbeat):
            return self._ack(cmd.seq, AckStatus.OK)
        if isinstance(cmd, Move):
            speed = min(
                cmd.speed_dps, self._lim.max_speed_dps, self._sim.servo_speed_dps
            )
            heapq.heappush(
                self._pending,
                _Pending(
                    self._t + self._sim.dead_time_s,
                    next(self._order),
                    self._lim.clamp_pan(cmd.pan_deg),
                    self._lim.clamp_tilt(cmd.tilt_deg),
                    speed,
                ),
            )
            return self._ack(cmd.seq, AckStatus.OK)
        return self._ack(cmd.seq, AckStatus.REJECTED)  # pragma: no cover


class SimulatedTransport:
    """Transporte que conecta el adaptador con el nodo simulado a través de un enlace."""

    kind = "simulated"

    def __init__(
        self, node: SimulatedPanTiltNode, clock: ManualClock, sim: SimulatorConfig
    ) -> None:
        self._node = node
        self._clock = clock
        self._sim = sim
        self._rng = random.Random(sim.seed + 1)
        self._downlink: list[tuple[float, int, bytes]] = []
        self._order = itertools.count()

    def _delay(self) -> float:
        jitter = (
            self._rng.uniform(0.0, self._sim.jitter_s) if self._sim.jitter_s else 0.0
        )
        return self._sim.one_way_latency_s + jitter

    def _lost(self) -> bool:
        return self._sim.drop_prob > 0 and self._rng.random() < self._sim.drop_prob

    def send(self, frame: bytes) -> None:
        if self._lost():
            return
        self._node.deliver(frame, self._clock.now() + self._delay())

    def pump(self) -> None:
        """Avanza el nodo hasta el instante actual y encola sus acks en el enlace."""
        for t, ack in self._node.advance_to(self._clock.now()):
            if not self._lost():
                heapq.heappush(
                    self._downlink, (t + self._delay(), next(self._order), ack)
                )

    def recv(self) -> list[bytes]:
        self.pump()
        now = self._clock.now()
        ready: list[bytes] = []
        while self._downlink and self._downlink[0][0] <= now:
            ready.append(heapq.heappop(self._downlink)[2])
        return ready

    def close(self) -> None:
        self._downlink.clear()
