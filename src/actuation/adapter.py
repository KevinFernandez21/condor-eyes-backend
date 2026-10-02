"""Adaptador Python del nodo Pan-Tilt: secuencias, acks, salud y metadata.

El adaptador es síncrono y no abre hilos: el llamador invoca :meth:`track` (o
:meth:`poll`) a la cadencia del lazo de control. Solo emite metadata
serializable (``MetadataEnvelope``); ningún frame pasa por aquí.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from collections.abc import Collection
from enum import StrEnum
from typing import Any, Protocol

from bus.hub import MetadataEnvelope, MetadataHub, Topic

from .config import ActuationConfig
from .controller import (
    ControlDecision,
    ControlState,
    TargetObservation,
    TrackingController,
)
from .protocol import (
    MS_MODULUS,
    SEQ_MODULUS,
    Ack,
    AckStatus,
    ClearEstop,
    Command,
    EmergencyStop,
    Heartbeat,
    Move,
    NodeState,
    ProtocolError,
    decode_ack,
    encode_command,
    seq_is_newer,
)
from .transport import Transport

_HISTORY = 200
_NOISE_WINDOW = 50
_NOISE_MIN_SAMPLES = 10
_MAX_PENDING = 256
_MAX_ENVELOPES = 1000
_DEFAULT_RELEASE_SOURCES = ("supervisor",)

_log = logging.getLogger(__name__)


class Clock(Protocol):
    def now(self) -> float: ...


class MonotonicClock:
    """Reloj real para producción (``time.monotonic``)."""

    def now(self) -> float:
        return time.monotonic()


class LinkHealth(StrEnum):
    UNKNOWN = "unknown"  # aún sin ningún ack
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # latencia alta o acks perdidos recientemente
    LOST = "lost"  # sin acks: el host no emite movimientos
    ESTOP = "estop"  # parada de emergencia enclavada


class _Controller(Protocol):
    last_error: float | None

    def update(
        self,
        observation: TargetObservation | None,
        now: float,
        measured: tuple[float, float] | None = None,
    ) -> ControlDecision: ...

    def resync(self, position: tuple[float, float]) -> None: ...


class PanTiltActuator:
    """Controla el nodo Pan-Tilt a través de un :class:`Transport`."""

    def __init__(
        self,
        transport: Transport,
        cfg: ActuationConfig,
        clock: Clock | None = None,
        camera_id: str = "cam-01",
        controller: _Controller | None = None,
        release_sources: Collection[str] = _DEFAULT_RELEASE_SOURCES,
    ) -> None:
        self._tp = transport
        self._cfg = cfg
        self._clock: Clock = clock or MonotonicClock()
        self._camera_id = camera_id
        self._ctl: _Controller = controller or TrackingController(cfg)
        self._simulated = transport.kind == "simulated"
        self._release_sources = frozenset(release_sources)
        # Un retroceso de node_ms mayor que esto se interpreta como reinicio del nodo.
        self._reboot_ms = max(500.0, cfg.comms.comms_timeout_s * 1000)

        t0 = self._clock.now()
        self._t0 = t0
        self._seq = 0
        self._last_send = t0
        self._last_move: tuple[float, float, float] | None = None
        self._pending: dict[int, float] = {}
        self._last_ack_time: float | None = None
        self._last_node_ms: int | None = None
        self._synced = False  # el nodo ya aceptó un comando con nuestra secuencia
        self._startup_pending = False
        self._adopt_floor: int | None = None
        self._estop_seqs: deque[int] = deque(maxlen=16)
        self._alarms: list[str] = []
        self._measured: tuple[float, float] | None = None
        self._node_state: NodeState | None = None
        self._latched = False
        self._estop: tuple[bytes, int, float, int] | None = (
            None  # trama, seq, t, reintentos
        )
        self._clear_seq: int | None = None
        self._health = LinkHealth.UNKNOWN
        self._last_report: float | None = None
        self._envelopes: list[tuple[Topic, MetadataEnvelope]] = []

        self._latencies: deque[float] = deque(maxlen=_HISTORY)
        self._errors: deque[float] = deque(maxlen=_HISTORY)
        self._idle_samples: deque[tuple[float, float]] = deque(maxlen=_NOISE_WINDOW)
        self._timeouts: deque[float] = deque(maxlen=_HISTORY)
        self._counters = {
            "moves_sent": 0,
            "heartbeats_sent": 0,
            "acks_ok": 0,
            "acks_duplicate": 0,
            "acks_stale": 0,
            "acks_rejected": 0,
            "acks_estop": 0,
            "ack_timeouts": 0,
            "bad_frames": 0,
            "unmatched_acks": 0,
            "moves_suppressed": 0,
            "estop_unconfirmed": 0,
            "estop_send_failures": 0,
            "node_reboots": 0,
            "seq_resyncs": 0,
            "rejected_commands": 0,
            "envelopes_dropped": 0,
        }

    # --- API pública ---

    @property
    def health(self) -> LinkHealth:
        return self._health

    @property
    def measured(self) -> tuple[float, float] | None:
        """Última posición (pan, tilt) reportada por el nodo, con su ruido."""
        return self._measured

    @property
    def node_state(self) -> NodeState | None:
        return self._node_state

    def start(self) -> None:
        """Arranque seguro: ordena el neutro a baja velocidad antes de seguir nada."""
        lim, ctl = self._cfg.limits, self._cfg.control
        self._startup_pending = True
        self._send_move(
            lim.neutral_pan_deg, lim.neutral_tilt_deg, ctl.neutral_speed_dps, "startup"
        )

    def track(self, observation: TargetObservation | None) -> ControlDecision:
        """Un ciclo de control: lee acks, decide, y envía solo si hay un cambio útil."""
        self.poll()
        now = self._clock.now()
        decision = self._ctl.update(observation, now, measured=self._measured)
        if self._ctl.last_error is not None:
            self._errors.append(self._ctl.last_error)
        if self._health in (LinkHealth.LOST, LinkHealth.ESTOP):
            self._counters["moves_suppressed"] += 1
            return decision
        lim = self._cfg.limits
        pan = lim.clamp_pan(decision.pan_deg)
        tilt = lim.clamp_tilt(decision.tilt_deg)
        speed = min(max(decision.speed_dps, 1e-3), lim.max_speed_dps)
        last = self._last_move
        eps = self._cfg.control.epsilon_deg
        if (
            last is None
            or abs(pan - last[0]) > eps
            or abs(tilt - last[1]) > eps
            or abs(speed - last[2]) > 1e-6
        ):
            self._send_move(pan, tilt, speed, decision.state)
        return decision

    def poll(self) -> None:
        """Procesa acks, vigila timeouts, mantiene el latido y actualiza la salud.

        Los fallos del transporte (``TransportError``/``OSError``) **se propagan**:
        el llamador decide cómo reaccionar. Mientras tanto no se envía nada, el
        watchdog del nodo (``node_watchdog_s``) lo congela en sitio y, si hubo un
        ESTOP, el enclavamiento del host se conserva.
        """
        now = self._clock.now()
        for frame in self._tp.recv():
            try:
                ack = decode_ack(frame)
            except ProtocolError:
                self._counters["bad_frames"] += 1
                continue
            self._on_ack(ack, now)
        self._expire(now)
        self._retry_estop(now)
        if now - self._last_send >= self._cfg.comms.heartbeat_interval_s:
            self._send(Heartbeat(self._next_seq()), now)
            self._counters["heartbeats_sent"] += 1
        self._update_health(now)

    def emergency_stop(self) -> None:
        """Detiene el nodo y bloquea todo movimiento hasta :meth:`release_estop`.

        El enclavamiento del host se activa **antes** de enviar: si el transporte
        falla, la excepción se propaga pero el host queda en ESTOP, no emite
        movimientos y reintenta la trama en cada :meth:`poll`.
        """
        now = self._clock.now()
        seq = self._next_seq()
        frame = encode_command(EmergencyStop(seq))
        self._latched = True
        self._clear_seq = None
        self._estop = (frame, seq, now, self._cfg.comms.estop_retries)
        self._estop_seqs.append(seq)
        self._set_health(LinkHealth.ESTOP, now)
        self._send_estop_frame(frame, now)

    def _send_estop_frame(self, frame: bytes, now: float) -> None:
        try:
            self._tp.send(frame)
        except Exception:
            self._counters["estop_send_failures"] += 1
            self._raise_alarm(
                "estop_send_failed", "no se pudo entregar la trama de ESTOP"
            )
            raise
        self._last_send = now

    def release_estop(self) -> None:
        """Pide al nodo liberar el ESTOP; el host reanuda al recibir su confirmación."""
        now = self._clock.now()
        seq = self._next_seq()
        self._send(ClearEstop(seq), now)
        self._clear_seq = seq

    def apply_command_envelope(self, envelope: MetadataEnvelope) -> None:
        """Atiende órdenes del bus; ignora el resto.

        ``ptz.estop`` se acepta de cualquier fuente (parar siempre es seguro).
        ``ptz.release_estop`` solo de las fuentes autorizadas (por defecto
        ``supervisor``). ``source`` es una etiqueta dentro del proceso, no una
        autenticación: la defensa real es que solo el supervisor publique en el bus.
        """
        kind = envelope.payload.get("kind")
        if kind == "ptz.estop":
            self.emergency_stop()
        elif kind == "ptz.release_estop":
            if envelope.source in self._release_sources:
                self.release_estop()
            else:
                self._counters["rejected_commands"] += 1
                _log.warning("release_estop rechazado de la fuente %r", envelope.source)

    def drain_envelopes(self) -> list[tuple[Topic, MetadataEnvelope]]:
        """Entrega y vacía la metadata pendiente de publicar."""
        out, self._envelopes = self._envelopes, []
        return out

    async def publish_to(self, hub: MetadataHub) -> None:
        """Publica la metadata pendiente a través de la interfaz tipada del bus."""
        for topic, envelope in self.drain_envelopes():
            await hub.publish(topic, envelope)

    def close(self) -> None:
        self._tp.close()

    def metrics(self) -> dict[str, Any]:
        """Métricas de latencia, error de seguimiento, ruido y salud (serializables)."""
        lat = sorted(self._latencies)
        latency: dict[str, Any] = {"samples": len(lat), "last": None, "mean": None}
        latency.update({"p95": None, "max": None})
        if lat:
            latency.update(
                last=self._latencies[-1] * 1000,
                mean=statistics.fmean(lat) * 1000,
                p95=lat[min(len(lat) - 1, int(0.95 * len(lat)))] * 1000,
                max=lat[-1] * 1000,
            )
        errs = self._errors
        return {
            "simulated": self._simulated,
            "transport": self._tp.kind,
            "health": self._health.value,
            "node_state": self._node_state.value if self._node_state else None,
            "measured": list(self._measured) if self._measured else None,
            "latency_ms": latency,
            "tracking_error": {
                "last": self._ctl.last_error,
                "mean": statistics.fmean(errs) if errs else None,
            },
            "position_noise_deg": self._noise(),
            "alarms": list(self._alarms),
            "counters": dict(self._counters),
        }

    # --- internos ---

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) % SEQ_MODULUS
        return self._seq

    def _send(self, cmd: Command, now: float) -> None:
        self._tp.send(encode_command(cmd))
        self._last_send = now
        self._pending[cmd.seq] = now
        if len(self._pending) > _MAX_PENDING:
            self._pending.pop(next(iter(self._pending)))

    def _send_move(
        self, pan: float, tilt: float, speed: float, state: ControlState | str
    ) -> None:
        lim = self._cfg.limits
        pan = lim.clamp_pan(pan)
        tilt = lim.clamp_tilt(tilt)
        speed = min(max(speed, 1e-3), lim.max_speed_dps)
        now = self._clock.now()
        seq = self._next_seq()
        self._send(Move(seq, pan, tilt, speed), now)
        self._last_move = (pan, tilt, speed)
        self._counters["moves_sent"] += 1
        self._push(
            Topic.EVENTS,
            self._envelope(
                {
                    "kind": "ptz.command",
                    "seq": seq,
                    "pan_deg": pan,
                    "tilt_deg": tilt,
                    "speed_dps": speed,
                    "state": str(state),
                }
            ),
        )

    def _push(self, topic: Topic, envelope: MetadataEnvelope) -> None:
        """Encola metadata acotada: si nadie drena, se descarta lo más viejo."""
        self._envelopes.append((topic, envelope))
        if len(self._envelopes) > _MAX_ENVELOPES:
            del self._envelopes[0]
            self._counters["envelopes_dropped"] += 1

    def _raise_alarm(self, reason: str, detail: str) -> None:
        """Alarma explícita y crítica: evento en el bus, log de error y estado en métricas."""
        _log.error("alarma de actuación %s: %s", reason, detail)
        if reason in self._alarms:
            return
        self._alarms.append(reason)
        self._push(
            Topic.EVENTS,
            self._envelope(
                {
                    "kind": "ptz.alarm",
                    "reason": reason,
                    "severity": "critical",
                    "detail": detail,
                }
            ),
        )

    def _envelope(self, payload: dict[str, Any]) -> MetadataEnvelope:
        payload["simulated"] = self._simulated
        return MetadataEnvelope(
            source="actuation", stream_id=self._camera_id, payload=payload
        )

    def _on_ack(self, ack: Ack, now: float) -> None:
        sent = self._pending.pop(ack.seq, None)
        if sent is None:
            self._counters["unmatched_acks"] += 1
        else:
            self._latencies.append(max(now - sent, 0.0))
        previous = self._health
        self._last_ack_time = now
        key = {
            AckStatus.OK: "acks_ok",
            AckStatus.DUPLICATE: "acks_duplicate",
            AckStatus.STALE: "acks_stale",
            AckStatus.REJECTED: "acks_rejected",
            AckStatus.ESTOP: "acks_estop",
        }[ack.status]
        self._counters[key] += 1

        self._update_telemetry(ack)
        if ack.status in (AckStatus.OK, AckStatus.DUPLICATE) and (
            ack.seq not in self._estop_seqs
        ):
            # El ESTOP se atiende sin validar el seq: no prueba que estemos sincronizados.
            self._synced = True
            self._startup_pending = False
        self._maybe_adopt_sequence(ack, sent is not None)

        if (
            self._estop is not None
            and ack.seq == self._estop[1]
            and ack.state is NodeState.ESTOP
        ):
            self._estop = None
        if self._clear_seq is not None and ack.seq == self._clear_seq:
            self._clear_seq = None
            if ack.state is not NodeState.ESTOP:
                self._latched = False
                self._resync()
        elif previous is LinkHealth.LOST and not self._latched:
            self._resync()

    def _update_telemetry(self, ack: Ack) -> None:
        """La telemetría solo avanza, salvo un reinicio del nodo (``ms`` vuelve atrás).

        ``node_ms`` es uint32 con vuelta modular. Un ack viejo y reordenado
        (retroceso pequeño) se ignora; un retroceso grande es un reinicio del
        nodo (brownout): se acepta su posición y se re-sincroniza el controlador.
        """
        last = self._last_node_ms
        if last is not None:
            forward = (ack.node_ms - last) % MS_MODULUS
            if forward >= MS_MODULUS // 2:  # el reloj del nodo retrocedió
                back = MS_MODULUS - forward
                if back <= self._reboot_ms:
                    return
                self._counters["node_reboots"] += 1
                self._last_move = None  # forzar reenvío del objetivo vigente
                self._measured = (ack.pan_deg, ack.tilt_deg)
                self._ctl.resync(self._measured)
        self._last_node_ms = ack.node_ms
        self._measured = (ack.pan_deg, ack.tilt_deg)
        self._node_state = ack.state
        if ack.state is NodeState.IDLE:
            self._idle_samples.append(self._measured)
        if ack.state is NodeState.ESTOP:
            self._latched = True

    def _maybe_adopt_sequence(self, ack: Ack, matched: bool) -> None:
        """Re-sincroniza la secuencia tras un reinicio del host.

        El nodo conserva su último ``seq`` y rechazaría como ``stale`` todo lo
        que el host reiniciado (``seq`` desde 0) envíe. Mientras el host no haya
        sido aceptado (``_synced`` falso), un ack ``stale`` con ``last_seq``
        sobre un comando que enviamos hace avanzar nuestro contador hasta ese
        valor. Nunca retrocede ni se ejecuta ya sincronizado, así que no debilita
        la protección contra repeticiones: el siguiente ``seq`` siempre es
        posterior al último aceptado por el nodo.
        """
        if (
            self._synced
            or not matched
            or ack.status is not AckStatus.STALE
            or ack.last_seq is None
        ):
            return
        if self._adopt_floor is not None and seq_is_newer(self._adopt_floor, ack.seq):
            return  # ack de un comando anterior a la adopción ya hecha
        self._seq = ack.last_seq
        self._counters["seq_resyncs"] += 1
        self._last_move = None
        pending_clear = self._clear_seq is not None and ack.seq == self._clear_seq
        if self._startup_pending:
            self.start()
        if pending_clear:
            self.release_estop()
        self._adopt_floor = self._seq

    def _resync(self) -> None:
        if self._measured is not None:
            self._ctl.resync(self._measured)

    def _expire(self, now: float) -> None:
        limit = self._cfg.comms.ack_timeout_s
        for seq, sent in list(self._pending.items()):
            if now - sent > limit:
                del self._pending[seq]
                self._counters["ack_timeouts"] += 1
                self._timeouts.append(now)

    def _retry_estop(self, now: float) -> None:
        if self._estop is None:
            return
        frame, seq, sent, retries = self._estop
        if now - sent < self._cfg.comms.ack_timeout_s:
            return
        if retries <= 0:
            self._estop = None
            self._counters["estop_unconfirmed"] += 1
            self._raise_alarm(
                "estop_unconfirmed",
                "el nodo no confirmó el ESTOP tras agotar los reintentos",
            )
            return
        self._estop = (frame, seq, now, retries - 1)
        self._send_estop_frame(frame, now)

    def _update_health(self, now: float) -> None:
        comms = self._cfg.comms
        reference = self._last_ack_time if self._last_ack_time is not None else self._t0
        if self._latched:
            health = LinkHealth.ESTOP
        elif now - reference > comms.comms_timeout_s:
            health = LinkHealth.LOST
        elif self._last_ack_time is None:
            health = LinkHealth.UNKNOWN
        elif (self._latencies and self._latencies[-1] > comms.degraded_latency_s) or (
            self._timeouts and now - self._timeouts[-1] <= comms.comms_timeout_s
        ):
            health = LinkHealth.DEGRADED
        else:
            health = LinkHealth.HEALTHY
        self._set_health(health, now)

    def _set_health(self, health: LinkHealth, now: float) -> None:
        changed = health is not self._health
        self._health = health
        due = (
            self._last_report is None
            or now - self._last_report >= self._cfg.comms.health_report_interval_s
        )
        if changed or due:
            self._last_report = now
            self._push(
                Topic.HEALTH, self._envelope({"kind": "ptz.health", **self.metrics()})
            )

    def _noise(self) -> dict[str, float] | None:
        samples = self._idle_samples
        if len(samples) < _NOISE_MIN_SAMPLES:
            return None
        return {
            "pan": statistics.pstdev(s[0] for s in samples),
            "tilt": statistics.pstdev(s[1] for s in samples),
        }


__all__ = [
    "Clock",
    "LinkHealth",
    "MonotonicClock",
    "PanTiltActuator",
]
