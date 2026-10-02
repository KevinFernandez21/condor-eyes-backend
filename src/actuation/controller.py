"""Convierte el error de seguimiento en espacio de imagen en comandos Pan-Tilt acotados.

Convenciones: el error está normalizado a [-1, 1] con origen en el centro de la
imagen; ``error_x > 0`` es objetivo a la derecha (pan positivo) y ``error_y > 0``
es objetivo por debajo del centro (tilt negativo, pues tilt positivo mira arriba).

Garantías (verificadas con pruebas de propiedades):

* el objetivo nunca sale de los límites mecánicos;
* el cambio de objetivo por ciclo nunca excede ``velocidad * dt``;
* una observación perdida, vieja o no finita congela el movimiento y, tras
  ``hold_s``, devuelve la cámara al neutro de forma gradual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .config import ActuationConfig

_MAX_DT_S = 0.5  # un ciclo detenido no autoriza un salto grande al reanudar


class ControlState(StrEnum):
    TRACKING = "tracking"
    HOLDING = "holding"  # objetivo perdido: congelado en sitio
    RETURNING = "returning"  # regresando al neutro
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class TargetObservation:
    """Error del track seleccionado respecto al centro de la imagen."""

    error_x: float
    error_y: float
    timestamp: float
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class ControlDecision:
    """Objetivo angular acotado y la velocidad con la que alcanzarlo."""

    pan_deg: float
    tilt_deg: float
    speed_dps: float
    state: ControlState


def _step_toward(current: float, target: float, max_step: float) -> float:
    return current + max(-max_step, min(max_step, target - current))


class TrackingController:
    """Controlador proporcional con zona muerta, suavizado y límite de velocidad."""

    def __init__(self, cfg: ActuationConfig) -> None:
        self._lim = cfg.limits
        self._ctl = cfg.control
        self.reset()

    def reset(self) -> None:
        """Vuelve al estado inicial (cámara en el neutro, sin historia)."""
        self._cmd = (self._lim.neutral_pan_deg, self._lim.neutral_tilt_deg)
        self._last_t: float | None = None
        self._ema: tuple[float, float] | None = None
        self._lost_since: float | None = None
        self.last_error: float | None = None

    @property
    def command(self) -> tuple[float, float]:
        """Último objetivo ordenado (pan, tilt)."""
        return self._cmd

    def resync(self, position: tuple[float, float]) -> None:
        """Alinea el objetivo ordenado con la posición real tras una interrupción.

        Se usa al liberar un ESTOP o recuperar el enlace: evita ordenar un salto
        desde un objetivo que la cámara nunca alcanzó.
        """
        self._cmd = self._base(position)
        self._ema = None
        self._lost_since = None
        self.last_error = None

    def update(
        self,
        observation: TargetObservation | None,
        now: float,
        measured: tuple[float, float] | None = None,
    ) -> ControlDecision:
        """Calcula el siguiente objetivo. ``measured`` es la posición reportada por el nodo."""
        dt = (
            1.0 / self._ctl.control_rate_hz
            if self._last_t is None
            else min(max(now - self._last_t, 0.0), _MAX_DT_S)
        )
        self._last_t = now

        if self._is_valid(observation, now):
            assert observation is not None
            return self._track(observation, dt, measured)
        return self._lost(now, dt, measured)

    # --- observación válida ---

    def _is_valid(self, obs: TargetObservation | None, now: float) -> bool:
        if obs is None:
            return False
        if not (
            math.isfinite(obs.error_x)
            and math.isfinite(obs.error_y)
            and math.isfinite(obs.timestamp)
        ):
            return False
        return now - obs.timestamp <= self._ctl.target_timeout_s

    def _track(
        self,
        obs: TargetObservation,
        dt: float,
        measured: tuple[float, float] | None,
    ) -> ControlDecision:
        ctl = self._ctl
        self._lost_since = None
        ex = min(max(obs.error_x, -1.0), 1.0)
        ey = min(max(obs.error_y, -1.0), 1.0)
        self.last_error = math.hypot(ex, ey)

        if self._ema is None:
            self._ema = (ex, ey)
        else:
            a = ctl.smoothing_alpha
            self._ema = (
                a * ex + (1 - a) * self._ema[0],
                a * ey + (1 - a) * self._ema[1],
            )
        sx, sy = self._ema
        if abs(sx) < ctl.deadband:
            sx = 0.0
        if abs(sy) < ctl.deadband:
            sy = 0.0

        base = self._base(measured)
        want_pan = self._lim.clamp_pan(base[0] + ctl.kp * sx * ctl.hfov_deg / 2)
        want_tilt = self._lim.clamp_tilt(base[1] - ctl.kp * sy * ctl.vfov_deg / 2)
        if sx == 0.0:
            want_pan = self._cmd[0]
        if sy == 0.0:
            want_tilt = self._cmd[1]

        max_step = ctl.max_speed_dps * dt
        self._cmd = (
            self._lim.clamp_pan(_step_toward(self._cmd[0], want_pan, max_step)),
            self._lim.clamp_tilt(_step_toward(self._cmd[1], want_tilt, max_step)),
        )
        return ControlDecision(*self._cmd, ctl.max_speed_dps, ControlState.TRACKING)

    def _base(self, measured: tuple[float, float] | None) -> tuple[float, float]:
        if measured is not None and all(math.isfinite(v) for v in measured):
            return (self._lim.clamp_pan(measured[0]), self._lim.clamp_tilt(measured[1]))
        return self._cmd

    # --- objetivo perdido ---

    def _lost(
        self, now: float, dt: float, measured: tuple[float, float] | None
    ) -> ControlDecision:
        ctl = self._ctl
        self._ema = None
        self.last_error = None
        if self._lost_since is None:
            # Primer ciclo sin objetivo: congelar donde está la cámara realmente.
            self._lost_since = now
            self._cmd = self._base(measured)
            return ControlDecision(*self._cmd, ctl.max_speed_dps, ControlState.HOLDING)

        neutral = (self._lim.neutral_pan_deg, self._lim.neutral_tilt_deg)
        if not ctl.return_to_neutral or now - self._lost_since < ctl.hold_s:
            return ControlDecision(*self._cmd, ctl.max_speed_dps, ControlState.HOLDING)
        if self._cmd == neutral:
            return ControlDecision(
                *self._cmd, ctl.neutral_speed_dps, ControlState.NEUTRAL
            )

        max_step = ctl.neutral_speed_dps * dt
        self._cmd = (
            _step_toward(self._cmd[0], neutral[0], max_step),
            _step_toward(self._cmd[1], neutral[1], max_step),
        )
        state = ControlState.NEUTRAL if self._cmd == neutral else ControlState.RETURNING
        return ControlDecision(*self._cmd, ctl.neutral_speed_dps, state)
