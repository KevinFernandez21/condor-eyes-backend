"""Carga de zonas y política desde TOML (`configs/events.toml`)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .model import AuthorizedPolicy, EventPolicy, Zone, as_points


def load_events_config(path: str | Path) -> tuple[list[Zone], EventPolicy]:
    with open(path, "rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    pol = dict(data.get("policy", {}))
    if "authorized" in pol:
        pol["authorized"] = AuthorizedPolicy(pol["authorized"])
    policy = EventPolicy(**pol)
    zones = []
    for z in data.get("zones", []):
        z = dict(z)
        z["polygon"] = as_points(z["polygon"])
        zones.append(Zone(**z))
    if not zones:
        raise ValueError(f"{path} no define [[zones]]")
    names = [z.name for z in zones]
    if len(names) != len(set(names)):
        raise ValueError("Nombres de zona duplicados")
    return zones, policy
