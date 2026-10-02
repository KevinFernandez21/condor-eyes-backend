"""Configuración del nodo Pan-Tilt: límites mecánicos, control, enlace y simulador."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    """Límites mecánicos y techo duro de velocidad (grados respecto al neutro)."""

    pan_min_deg: float = -80.0
    pan_max_deg: float = 80.0
    tilt_min_deg: float = -30.0
    tilt_max_deg: float = 45.0
    neutral_pan_deg: float = 0.0
    neutral_tilt_deg: float = 0.0
    max_speed_dps: float = 60.0

    def clamp_pan(self, value: float) -> float:
        return min(max(value, self.pan_min_deg), self.pan_max_deg)

    def clamp_tilt(self, value: float) -> float:
        return min(max(value, self.tilt_min_deg), self.tilt_max_deg)


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """Parámetros del lazo error de imagen -> comando."""

    hfov_deg: float = 90.0
    vfov_deg: float = 55.0
    kp: float = 0.6
    deadband: float = 0.04
    smoothing_alpha: float = 0.5
    max_speed_dps: float = 45.0
    control_rate_hz: float = 20.0
    target_timeout_s: float = 0.5
    hold_s: float = 1.0
    return_to_neutral: bool = True
    neutral_speed_dps: float = 20.0
    epsilon_deg: float = 0.05


@dataclass(frozen=True, slots=True)
class CommsConfig:
    """Tiempos del enlace host <-> nodo."""

    ack_timeout_s: float = 0.25
    comms_timeout_s: float = 1.0
    node_watchdog_s: float = 0.75
    heartbeat_interval_s: float = 0.25
    estop_retries: int = 5
    degraded_latency_s: float = 0.15


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    """Parámetros del simulador (supuestos, no mediciones de hardware)."""

    one_way_latency_s: float = 0.01
    jitter_s: float = 0.0
    dead_time_s: float = 0.03
    noise_std_deg: float = 0.1
    drop_prob: float = 0.0
    servo_speed_dps: float = 120.0
    seed: int = 1234
    physics_dt_s: float = 0.002


@dataclass(frozen=True, slots=True)
class ActuationConfig:
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)

    def __post_init__(self) -> None:
        _validate(self)


_SECTIONS: dict[str, type] = {
    "limits": LimitsConfig,
    "control": ControlConfig,
    "comms": CommsConfig,
    "simulator": SimulatorConfig,
}


def _validate(cfg: ActuationConfig) -> None:
    lim, ctl, com, sim = cfg.limits, cfg.control, cfg.comms, cfg.simulator
    for section in (lim, ctl, com, sim):
        for f in fields(section):
            value = getattr(section, f.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{f.name} debe ser finito")
    if not lim.pan_min_deg < lim.neutral_pan_deg < lim.pan_max_deg:
        raise ValueError("el neutro de pan debe quedar dentro de (pan_min, pan_max)")
    if not lim.tilt_min_deg < lim.neutral_tilt_deg < lim.tilt_max_deg:
        raise ValueError("el neutro de tilt debe quedar dentro de (tilt_min, tilt_max)")
    if lim.max_speed_dps <= 0:
        raise ValueError("limits.max_speed_dps debe ser positivo")
    if not 0 < ctl.max_speed_dps <= lim.max_speed_dps:
        raise ValueError(
            "control.max_speed_dps debe estar en (0, limits.max_speed_dps]"
        )
    if not 0 < ctl.neutral_speed_dps <= lim.max_speed_dps:
        raise ValueError(
            "control.neutral_speed_dps debe estar en (0, limits.max_speed_dps]"
        )
    if not 0 < ctl.kp <= 1.0:
        raise ValueError("control.kp debe estar en (0, 1]")
    if not 0 <= ctl.deadband < 1:
        raise ValueError("control.deadband debe estar en [0, 1)")
    if not 0 < ctl.smoothing_alpha <= 1:
        raise ValueError("control.smoothing_alpha debe estar en (0, 1]")
    if not (0 < ctl.hfov_deg < 180 and 0 < ctl.vfov_deg < 180):
        raise ValueError("hfov_deg y vfov_deg deben estar en (0, 180)")
    if ctl.control_rate_hz <= 0 or ctl.target_timeout_s <= 0 or ctl.hold_s < 0:
        raise ValueError("control_rate_hz y target_timeout_s deben ser positivos")
    if ctl.epsilon_deg < 0:
        raise ValueError("control.epsilon_deg no puede ser negativo")
    if com.ack_timeout_s <= 0 or com.heartbeat_interval_s <= 0:
        raise ValueError("los tiempos de comms deben ser positivos")
    if com.comms_timeout_s < com.ack_timeout_s:
        raise ValueError("comms_timeout_s debe ser >= ack_timeout_s")
    if com.node_watchdog_s <= com.heartbeat_interval_s:
        raise ValueError("node_watchdog_s debe ser mayor que heartbeat_interval_s")
    if com.estop_retries < 0:
        raise ValueError("estop_retries no puede ser negativo")
    if not 0 <= sim.drop_prob <= 1:
        raise ValueError("simulator.drop_prob debe estar en [0, 1]")
    if sim.servo_speed_dps <= 0 or sim.physics_dt_s <= 0:
        raise ValueError("servo_speed_dps y physics_dt_s deben ser positivos")
    if min(sim.one_way_latency_s, sim.jitter_s, sim.dead_time_s, sim.noise_std_deg) < 0:
        raise ValueError("los parámetros del simulador no pueden ser negativos")


def load_actuation_config(path: str | Path | None = None) -> ActuationConfig:
    """Carga la tabla ``[pan_tilt]`` de un TOML; sin ruta devuelve los valores por defecto."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f).get("pan_tilt", {})
    unknown_sections = set(data) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError(
            f"Sección desconocida en [pan_tilt]: {sorted(unknown_sections)}"
        )
    built: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        values = data.get(name, {})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"Claves desconocidas en [pan_tilt.{name}]: {sorted(unknown)}"
            )
        built[name] = cls(**values)
    return ActuationConfig(**built)
