"""Carga y validación estricta de ``configs/system.toml``.

Una clave desconocida, un número no finito (``nan``/``inf``), un booleano donde
se espera un número o un ``kind`` fuera de catálogo es un ``ConfigError``: el
sistema no arranca con una configuración ambigua.

Las rutas relativas (config de fusión, video, JSONL) se resuelven contra la
raíz del repositorio (padre del directorio del TOML), de modo que el comando
funciona desde cualquier directorio de trabajo.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "system.toml"

CAMERA_KINDS = ("fake", "webcam", "file")
DETECTOR_KINDS = ("fake", "moving", "yolov8n", "none")
TAG_KINDS = ("sim", "c6", "replay", "none")
IDENTITY_KINDS = ("sim", "none")
ACTUATOR_KINDS = ("sim", "none")


class ConfigError(ValueError):
    """La configuración del sistema es inválida."""


# --- secciones --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CameraConfig:
    kind: str = "fake"
    stream_id: str = "cam-01"
    fps: float = 10.0
    device_index: int = 0  # webcam
    path: str = ""  # file: ruta del video grabado


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    kind: str = "fake"
    weights: str = ""  # yolov8n: sobreescribe el modelo de surveillance.toml
    config_path: str = "configs/surveillance.toml"


@dataclass(frozen=True, slots=True)
class TagConfig:
    kind: str = "sim"
    path: str = ""  # replay: JSONL con estimaciones de ubicación
    interval_s: float = 1.0


@dataclass(frozen=True, slots=True)
class IdentityConfig:
    kind: str = "sim"
    interval_s: float = 2.0


@dataclass(frozen=True, slots=True)
class ActuatorConfig:
    kind: str = "sim"
    interval_s: float = 1.0


@dataclass(frozen=True, slots=True)
class ZoneConfig:
    """Zona rectangular en fracciones horizontales del encuadre ([x_min, x_max))."""

    zone_id: str
    restricted: bool = True
    x_min: float = 0.0
    x_max: float = 1.0


@dataclass(frozen=True, slots=True)
class _SystemTable:
    """Escalares de ``[system]`` (se validan igual que el resto de secciones)."""

    heartbeat_interval_s: float = 2.0
    fusion_interval_s: float = 1.0
    shutdown_timeout_s: float = 5.0
    queue_size: int = 1024
    history_size: int = 1000
    recent_limit: int = 50
    fusion_config: str = "configs/fusion.toml"


@dataclass(frozen=True, slots=True)
class SystemConfig:
    profile: str
    heartbeat_interval_s: float = 2.0
    fusion_interval_s: float = 1.0
    shutdown_timeout_s: float = 5.0
    queue_size: int = 1024
    history_size: int = 1000
    recent_limit: int = 50
    fusion_config: str = "configs/fusion.toml"
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tag: TagConfig = field(default_factory=TagConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    zones: tuple[ZoneConfig, ...] = ()
    permissions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


# --- validación -------------------------------------------------------------

_POSITIVE = {
    "heartbeat_interval_s",
    "fusion_interval_s",
    "shutdown_timeout_s",
    "queue_size",
    "history_size",
    "recent_limit",
    "fps",
    "interval_s",
}
_MAXIMUM = {
    "heartbeat_interval_s": 3600.0,
    "fusion_interval_s": 3600.0,
    "shutdown_timeout_s": 600.0,
    "queue_size": 1_000_000,
    "history_size": 1_000_000,
    "recent_limit": 100_000,
    "fps": 240.0,
    "interval_s": 3600.0,
    "device_index": 64,
}
_KINDS = {
    "camera": CAMERA_KINDS,
    "detector": DETECTOR_KINDS,
    "tag": TAG_KINDS,
    "identity": IDENTITY_KINDS,
    "actuator": ACTUATOR_KINDS,
}


def _check_value(section: str, key: str, value: Any, expected: type) -> Any:
    where = f"[{section}].{key}"
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where} debe ser booleano")
        return value
    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} debe ser texto")
        return value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where} debe ser numérico")
    try:
        number = float(value)  # un entero enorme desborda aquí, no en isfinite
    except OverflowError:
        raise ConfigError(f"{where} es demasiado grande (máximo permitido)") from None
    if not math.isfinite(number):
        raise ConfigError(f"{where} debe ser un número finito")
    if key in _POSITIVE and number <= 0:
        raise ConfigError(f"{where} debe ser positivo")
    if key in _MAXIMUM and number > _MAXIMUM[key]:
        raise ConfigError(f"{where} supera el máximo permitido ({_MAXIMUM[key]})")
    if expected is int:
        if isinstance(value, float):
            if not value.is_integer():
                raise ConfigError(f"{where} debe ser entero")
            value = int(value)
        return value
    return float(value)


def _build(cls: type, section: str, table: Mapping[str, Any]) -> Any:
    """Instancia una dataclass desde un dict rechazando claves desconocidas."""
    specs = {f.name: f for f in fields(cls) if f.init}
    unknown = sorted(set(table) - set(specs))
    if unknown:
        raise ConfigError(f"Claves desconocidas en [{section}]: {unknown}")
    values: dict[str, Any] = {}
    for key, raw in table.items():
        annotation = specs[key].type
        expected = {"float": float, "int": int, "str": str, "bool": bool}.get(
            str(annotation)
        )
        if expected is None:
            raise ConfigError(
                f"[{section}].{key} no es configurable"
            )  # pragma: no cover
        values[key] = _check_value(section, key, raw, expected)
    kinds = _KINDS.get(section.rsplit(".", 1)[-1])
    kind = values.get("kind")
    if kinds is not None and kind is not None and kind not in kinds:
        raise ConfigError(
            f"[{section}].kind inválido: {kind!r} (permitidos: {', '.join(kinds)})"
        )
    return cls(**values)


def _zones(raw: Any) -> tuple[ZoneConfig, ...]:
    if not isinstance(raw, list):
        raise ConfigError("[[zones]] debe ser una lista de tablas")
    zones: list[ZoneConfig] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ConfigError("[[zones]] debe ser una lista de tablas")
        if "zone_id" not in item:
            raise ConfigError("Toda zona requiere zone_id")
        zone: ZoneConfig = _build(ZoneConfig, "zones", item)
        if not zone.zone_id.strip():
            raise ConfigError("zone_id no puede estar vacío")
        if zone.zone_id in seen:
            raise ConfigError(f"Zona duplicada: {zone.zone_id!r}")
        if not (0.0 <= zone.x_min < zone.x_max <= 1.0):
            raise ConfigError(
                f"Zona {zone.zone_id!r}: x_min/x_max inválidos (se exige 0 <= x_min < x_max <= 1)"
            )
        seen.add(zone.zone_id)
        zones.append(zone)
    return tuple(zones)


def _permissions(raw: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        raise ConfigError("[permissions] debe ser una tabla")
    out: dict[str, tuple[str, ...]] = {}
    for person, zones in raw.items():
        if not isinstance(zones, list) or not all(isinstance(z, str) for z in zones):
            raise ConfigError(
                f"[permissions].{person} debe ser una lista de zone_id (texto)"
            )
        out[str(person)] = tuple(zones)
    return out


def _resolve(base: Path, value: str) -> str:
    if not value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else base / path)


_SECTIONS = {
    "camera": CameraConfig,
    "detector": DetectorConfig,
    "tag": TagConfig,
    "identity": IdentityConfig,
    "actuator": ActuatorConfig,
}


def load_system_config(
    path: str | Path | None = None, profile: str | None = None
) -> SystemConfig:
    """Carga el TOML y devuelve la configuración del perfil pedido."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"No se encontró la configuración: {config_path}")
    try:
        with open(config_path, "rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"TOML inválido en {config_path}: {exc}") from exc

    unknown_top = sorted(set(raw) - {"system", "zones", "permissions", "profiles"})
    if unknown_top:
        raise ConfigError(f"Secciones desconocidas: {unknown_top}")

    system_table = dict(raw.get("system", {}))
    default_profile = system_table.pop("default_profile", "sim")
    if not isinstance(default_profile, str):
        raise ConfigError("[system].default_profile debe ser texto")
    system: _SystemTable = _build(_SystemTable, "system", system_table)
    profiles = raw.get("profiles", {})
    chosen = profile or default_profile
    if chosen not in profiles:
        known = ", ".join(sorted(profiles)) or "ninguno"
        raise ConfigError(f"Perfil desconocido: {chosen!r} (disponibles: {known})")
    table = profiles[chosen]
    if not isinstance(table, Mapping):
        raise ConfigError(f"[profiles.{chosen}] debe ser una tabla")
    unknown = sorted(set(table) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"Claves desconocidas en [profiles.{chosen}]: {unknown}")
    sections = {
        name: _build(cls, f"profiles.{chosen}.{name}", table.get(name, {}))
        for name, cls in _SECTIONS.items()
    }

    base = config_path.resolve().parent.parent
    sections["camera"] = _replace_paths(sections["camera"], base, ("path",))
    sections["tag"] = _replace_paths(sections["tag"], base, ("path",))
    sections["detector"] = _replace_paths(
        sections["detector"], base, ("weights", "config_path")
    )
    return SystemConfig(
        profile=chosen,
        heartbeat_interval_s=system.heartbeat_interval_s,
        fusion_interval_s=system.fusion_interval_s,
        shutdown_timeout_s=system.shutdown_timeout_s,
        queue_size=system.queue_size,
        history_size=system.history_size,
        recent_limit=system.recent_limit,
        fusion_config=_resolve(base, system.fusion_config),
        zones=_zones(raw.get("zones", [])),
        permissions=_permissions(raw.get("permissions", {})),
        **sections,
    )


def _replace_paths(section: Any, base: Path, names: tuple[str, ...]) -> Any:
    from dataclasses import replace

    return replace(section, **{n: _resolve(base, getattr(section, n)) for n in names})
