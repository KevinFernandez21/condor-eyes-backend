"""Carga y validación de `configs/location.toml`."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """La configuración de localización es inválida."""


@dataclass(frozen=True, slots=True)
class ZoneConfig:
    """Zona física con sus nodos ESP32-S3 y zonas vecinas."""

    zone_id: str
    nodes: tuple[str, ...]
    neighbors: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class LocationConfig:
    """Parámetros de frescura, suavizado, validación y confianza."""

    zones: Mapping[str, ZoneConfig]
    max_age_s: float = 10.0
    future_tolerance_s: float = 2.0
    evidence_max_age_s: float = 15.0
    node_timeout_s: float = 30.0
    retention_s: float = 300.0
    smoothing_window_s: float = 8.0
    smoothing_alpha: float = 0.5
    hysteresis_db: float = 4.0
    rssi_min_dbm: int = -127
    rssi_max_dbm: int = 0
    transition_min_s: float = 5.0
    transition_min_rssi_dbm: float = -80.0
    rssi_floor_dbm: float = -95.0
    rssi_strong_dbm: float = -60.0
    margin_full_db: float = 12.0
    samples_full: int = 3
    outage_penalty: float = 0.8
    battery_low_pct: int = 20
    _node_zone: Mapping[str, str] = field(default_factory=dict, repr=False)

    def zone_of_node(self, node_id: str) -> str | None:
        """Zona a la que pertenece un nodo, o None si no está configurado."""
        return self._node_zone.get(node_id)

    def nodes_of_zone(self, zone_id: str) -> tuple[str, ...]:
        return self.zones[zone_id].nodes

    def are_adjacent(self, a: str, b: str) -> bool:
        """Misma zona o vecinas (la relación es simétrica)."""
        if a == b:
            return True
        return b in self.zones[a].neighbors or a in self.zones[b].neighbors


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"la sección [{name}] debe ser una tabla")
    return value


def _number(section: Mapping[str, Any], name: str, key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"[{name}] {key} debe ser numérico")
    return float(value)


def parse_config(raw: Mapping[str, Any]) -> LocationConfig:
    """Construye y valida la configuración desde un dict (TOML ya parseado)."""
    zones_raw = raw.get("zones")
    if not isinstance(zones_raw, Mapping) or not zones_raw:
        raise ConfigError(
            "se requiere al menos una zona (sección de zonas) en [zones.<id>]"
        )

    zones: dict[str, ZoneConfig] = {}
    node_zone: dict[str, str] = {}
    for zone_id, zone_raw in zones_raw.items():
        nodes = tuple(zone_raw.get("nodes", ()))
        if not nodes:
            raise ConfigError(f"la zona '{zone_id}' debe tener al menos un nodo")
        for node in nodes:
            if node in node_zone:
                raise ConfigError(
                    f"el nodo '{node}' está en más de una zona "
                    f"('{node_zone[node]}' y '{zone_id}')"
                )
            node_zone[node] = zone_id
        zones[zone_id] = ZoneConfig(
            zone_id, nodes, frozenset(zone_raw.get("neighbors", ()))
        )
    for zone in zones.values():
        for neighbor in zone.neighbors:
            if neighbor not in zones:
                raise ConfigError(
                    f"la zona '{zone.zone_id}' declara una zona vecina "
                    f"inexistente: '{neighbor}'"
                )

    fresh = _section(raw, "freshness")
    smooth = _section(raw, "smoothing")
    valid = _section(raw, "validation")
    conf = _section(raw, "confidence")
    batt = _section(raw, "battery")
    defaults = LocationConfig(zones={})

    cfg = LocationConfig(
        zones=zones,
        max_age_s=_number(fresh, "freshness", "max_age_s", defaults.max_age_s),
        future_tolerance_s=_number(
            fresh, "freshness", "future_tolerance_s", defaults.future_tolerance_s
        ),
        evidence_max_age_s=_number(
            fresh, "freshness", "evidence_max_age_s", defaults.evidence_max_age_s
        ),
        node_timeout_s=_number(
            fresh, "freshness", "node_timeout_s", defaults.node_timeout_s
        ),
        retention_s=_number(fresh, "freshness", "retention_s", defaults.retention_s),
        smoothing_window_s=_number(
            smooth, "smoothing", "window_s", defaults.smoothing_window_s
        ),
        smoothing_alpha=_number(smooth, "smoothing", "alpha", defaults.smoothing_alpha),
        hysteresis_db=_number(
            smooth, "smoothing", "hysteresis_db", defaults.hysteresis_db
        ),
        rssi_min_dbm=int(
            _number(valid, "validation", "rssi_min_dbm", defaults.rssi_min_dbm)
        ),
        rssi_max_dbm=int(
            _number(valid, "validation", "rssi_max_dbm", defaults.rssi_max_dbm)
        ),
        transition_min_s=_number(
            valid, "validation", "transition_min_s", defaults.transition_min_s
        ),
        transition_min_rssi_dbm=_number(
            valid,
            "validation",
            "transition_min_rssi_dbm",
            defaults.transition_min_rssi_dbm,
        ),
        rssi_floor_dbm=_number(
            conf, "confidence", "rssi_floor_dbm", defaults.rssi_floor_dbm
        ),
        rssi_strong_dbm=_number(
            conf, "confidence", "rssi_strong_dbm", defaults.rssi_strong_dbm
        ),
        margin_full_db=_number(
            conf, "confidence", "margin_full_db", defaults.margin_full_db
        ),
        samples_full=int(
            _number(conf, "confidence", "samples_full", defaults.samples_full)
        ),
        outage_penalty=_number(
            conf, "confidence", "outage_penalty", defaults.outage_penalty
        ),
        battery_low_pct=int(
            _number(batt, "battery", "low_pct", defaults.battery_low_pct)
        ),
        _node_zone=node_zone,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: LocationConfig) -> None:
    for key in (
        "max_age_s",
        "evidence_max_age_s",
        "node_timeout_s",
        "retention_s",
        "smoothing_window_s",
        "transition_min_s",
        "margin_full_db",
    ):
        if getattr(cfg, key) <= 0:
            raise ConfigError(f"{key} debe ser mayor que 0")
    for key in ("future_tolerance_s", "hysteresis_db"):
        if getattr(cfg, key) < 0:
            raise ConfigError(f"{key} no puede ser negativo")
    if not 0 < cfg.smoothing_alpha <= 1:
        raise ConfigError("alpha debe estar en el intervalo (0, 1]")
    if cfg.smoothing_window_s > cfg.evidence_max_age_s:
        raise ConfigError("window_s no puede superar evidence_max_age_s")
    if cfg.rssi_floor_dbm >= cfg.rssi_strong_dbm:
        raise ConfigError("rssi_floor_dbm debe ser menor que rssi_strong_dbm")
    if cfg.rssi_min_dbm >= cfg.rssi_max_dbm:
        raise ConfigError("rssi_min_dbm debe ser menor que rssi_max_dbm")
    if not 0 < cfg.outage_penalty <= 1:
        raise ConfigError("outage_penalty debe estar en el intervalo (0, 1]")
    if cfg.samples_full < 1:
        raise ConfigError("samples_full debe ser al menos 1")
    if not 0 <= cfg.battery_low_pct <= 100:
        raise ConfigError("low_pct debe estar entre 0 y 100")


def load_config(path: str | Path) -> LocationConfig:
    """Carga la configuración desde un archivo TOML."""
    with Path(path).open("rb") as handle:
        return parse_config(tomllib.load(handle))
