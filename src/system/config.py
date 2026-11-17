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
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "system.toml"

CAMERA_KINDS = ("fake", "webcam", "file")
DETECTOR_KINDS = ("fake", "moving", "scenes", "yolov8n", "none")
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
class SceneConfig:
    """Parámetros de las escenas sintéticas del perfil ``sim`` (semilla: CLI)."""

    spawn_rate_per_s: float = 0.35  # altas por segundo y cámara
    max_actors: int = 4  # simultáneos por cámara
    vehicle_ratio: float = 0.25
    motion_ratio: float = 0.1  # movimiento sin clase
    stranger_ratio: float = 0.2  # persona sin tag ni rostro conocido
    no_tag_ratio: float = 0.15  # persona conocida sin tag
    dropouts: bool = True
    dropout_interval_s: float = 30.0  # media entre caídas de cámara
    dropout_duration_s: float = 6.0


DEFAULT_ORIGINS = ("http://127.0.0.1:8080", "http://localhost:8080")


@dataclass(frozen=True, slots=True)
class ApiConfig:
    """API de observabilidad (#42). El token nunca vive en el archivo (CLI/entorno)."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    allowed_origins: tuple[str, ...] = DEFAULT_ORIGINS


@dataclass(frozen=True, slots=True)
class ZoneConfig:
    """Zona rectangular en fracciones horizontales del encuadre ([x_min, x_max))."""

    zone_id: str
    restricted: bool = True
    x_min: float = 0.0
    x_max: float = 1.0


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """Cámara de la maqueta: identidad visible, resolución y zonas de su escena."""

    camera_id: str
    name: str
    scene: str = ""
    fps: float = 10.0
    width: int = 1920
    height: int = 1080
    zones: tuple[ZoneConfig, ...] = ()


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
    scenes: SceneConfig = field(default_factory=SceneConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    cameras: tuple[CameraSpec, ...] = ()
    zones: tuple[ZoneConfig, ...] = ()
    permissions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def multi_camera(self) -> bool:
        """Cámaras sintéticas múltiples (``camera.kind = "fake"`` con ``[[cameras]]``)."""
        return self.camera.kind == "fake" and bool(self.cameras)

    def camera_specs(self) -> tuple[CameraSpec, ...]:
        """Cámaras activas: las de la maqueta o la única del perfil."""
        if self.multi_camera:
            return self.cameras
        cam = self.camera
        return (
            CameraSpec(
                camera_id=cam.stream_id,
                name=cam.stream_id,
                scene=cam.kind,
                fps=cam.fps,
                width=0,
                height=0,
                zones=self.zones,
            ),
        )

    def effective_zones(self) -> tuple[ZoneConfig, ...]:
        """Zonas que usa la fusión: las de las cámaras de la maqueta o las globales."""
        if self.multi_camera:
            return tuple(z for c in self.cameras for z in c.zones)
        return self.zones


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
    "spawn_rate_per_s",
    "max_actors",
    "dropout_interval_s",
    "dropout_duration_s",
    "width",
    "height",
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
    "spawn_rate_per_s": 20.0,
    "max_actors": 50,
    "dropout_interval_s": 3600.0,
    "dropout_duration_s": 3600.0,
    "width": 8192,
    "height": 8192,
    "port": 65535,
}
_KINDS = {
    "camera": CAMERA_KINDS,
    "detector": DETECTOR_KINDS,
    "tag": TAG_KINDS,
    "identity": IDENTITY_KINDS,
    "actuator": ACTUATOR_KINDS,
}


def _scenes(table: Mapping[str, Any], section: str) -> SceneConfig:
    scenes: SceneConfig = _build(SceneConfig, section, table)
    for key in ("vehicle_ratio", "motion_ratio", "stranger_ratio", "no_tag_ratio"):
        if not 0.0 <= getattr(scenes, key) <= 1.0:
            raise ConfigError(f"[{section}].{key} debe estar en [0, 1]")
    if scenes.vehicle_ratio + scenes.motion_ratio > 1.0:
        raise ConfigError(
            f"[{section}]: vehicle_ratio + motion_ratio no puede superar 1"
        )
    if scenes.stranger_ratio + scenes.no_tag_ratio > 1.0:
        raise ConfigError(
            f"[{section}]: stranger_ratio + no_tag_ratio no puede superar 1"
        )
    return scenes


def _cameras(raw: Any) -> tuple[CameraSpec, ...]:
    if not isinstance(raw, list):
        raise ConfigError("[[cameras]] debe ser una lista de tablas")
    cameras: list[CameraSpec] = []
    seen: set[str] = set()
    zone_ids: set[str] = set()
    for item in raw:
        if (
            not isinstance(item, Mapping)
            or "camera_id" not in item
            or "name" not in item
        ):
            raise ConfigError("Toda cámara requiere camera_id y name")
        table = dict(item)
        zones = _zones(table.pop("zones", []))
        spec: CameraSpec = _build(
            CameraSpec, "cameras", {k: v for k, v in table.items()}
        )
        if not spec.camera_id.strip() or not spec.name.strip():
            raise ConfigError("camera_id y name no pueden estar vacíos")
        if spec.camera_id in seen:
            raise ConfigError(f"Cámara duplicada: {spec.camera_id!r}")
        seen.add(spec.camera_id)
        for zone in zones:
            if zone.zone_id in zone_ids:
                raise ConfigError(f"Zona duplicada entre cámaras: {zone.zone_id!r}")
            zone_ids.add(zone.zone_id)
        cameras.append(replace(spec, zones=zones))
    return tuple(cameras)


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


def _api(raw: Any) -> ApiConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError("[api] debe ser una tabla")
    table = dict(raw)
    origins = table.pop("allowed_origins", None)
    scalars = {k: v for k, v in table.items()}
    unknown = sorted(set(scalars) - {"enabled", "host", "port"})
    if unknown:
        raise ConfigError(f"Claves desconocidas en [api]: {unknown}")
    values: dict[str, Any] = {
        "enabled": _check_value("api", "enabled", scalars.get("enabled", False), bool),
        "host": _check_value("api", "host", scalars.get("host", "127.0.0.1"), str),
        "port": _check_value("api", "port", scalars.get("port", 8000), int),
    }
    if values["port"] < 0:
        raise ConfigError("[api].port inválido: el puerto no puede ser negativo")
    if not values["host"].strip():
        raise ConfigError("[api].host no puede estar vacío")
    if origins is None:
        values["allowed_origins"] = DEFAULT_ORIGINS
    else:
        if not isinstance(origins, list):
            raise ConfigError("[api].allowed_origins debe ser una lista de orígenes")
        if not all(isinstance(o, str) and o.strip() for o in origins):
            raise ConfigError("[api].allowed_origins debe contener solo texto no vacío")
        values["allowed_origins"] = tuple(origins)
    return ApiConfig(**values)


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
    "scenes": SceneConfig,
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

    unknown_top = sorted(
        set(raw) - {"system", "zones", "permissions", "profiles", "api", "cameras"}
    )
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
    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        where = f"profiles.{chosen}.{name}"
        sections[name] = (
            _scenes(table.get(name, {}), where)
            if cls is SceneConfig
            else _build(cls, where, table.get(name, {}))
        )

    if sections["detector"].kind == "scenes" and not (
        sections["camera"].kind == "fake" and raw.get("cameras")
    ):
        raise ConfigError(
            f'[profiles.{chosen}.detector] kind = "scenes" requiere '
            'camera.kind = "fake" y al menos una [[cameras]]'
        )

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
        api=_api(raw.get("api", {})),
        cameras=_cameras(raw.get("cameras", [])),
        zones=_zones(raw.get("zones", [])),
        permissions=_permissions(raw.get("permissions", {})),
        **sections,
    )


def _replace_paths(section: Any, base: Path, names: tuple[str, ...]) -> Any:
    from dataclasses import replace

    return replace(section, **{n: _resolve(base, getattr(section, n)) for n in names})
