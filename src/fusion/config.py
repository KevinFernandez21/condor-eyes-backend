"""Umbrales y ventanas de la fusión de evidencia (ver configs/fusion.toml)."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FusionConfig:
    """Parámetros calibrables. Los valores por defecto coinciden con configs/fusion.toml."""

    # Diferencia máxima entre el instante de la pista y el de la evidencia con que se correlaciona.
    time_window_s: float = 5.0
    # Antigüedad máxima (respecto a `now`) antes de considerar la evidencia obsoleta.
    track_max_age_s: float = 3.0
    identity_max_age_s: float = 10.0
    location_max_age_s: float = 15.0
    reid_max_age_s: float = 30.0
    # Tolerancia a relojes desfasados: evidencia fechada más allá de now + skew es inválida.
    max_clock_skew_s: float = 2.0
    # Confianzas mínimas para que cada fuente cuente como evidencia utilizable.
    min_track_confidence: float = 0.5
    min_identity_score: float = 0.65
    min_location_confidence: float = 0.6
    min_reid_confidence: float = 0.7
    # Confianza mínima del conjunto para poder corroborar (mínimo de las fuentes).
    min_decision_confidence: float = 0.6

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{f.name} debe ser numérico")
            if not math.isfinite(value):
                raise ValueError(f"{f.name} debe ser un número finito")
        for key in _RANGE_KEYS:
            if not 0.0 <= getattr(self, key) <= 1.0:
                raise ValueError(f"{key} debe estar en [0, 1]")
        for key in _POSITIVE_KEYS:
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} debe ser positivo")
        if self.max_clock_skew_s < 0:
            raise ValueError("max_clock_skew_s no puede ser negativo")


_RANGE_KEYS = (
    "min_track_confidence",
    "min_identity_score",
    "min_location_confidence",
    "min_reid_confidence",
    "min_decision_confidence",
)
_POSITIVE_KEYS = (
    "time_window_s",
    "track_max_age_s",
    "identity_max_age_s",
    "location_max_age_s",
    "reid_max_age_s",
)


def load_fusion_config(
    path: str | Path | None = None, **overrides: Any
) -> FusionConfig:
    """Carga la sección [fusion] de un TOML y aplica los overrides no nulos."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f).get("fusion", {})
    known = {f.name for f in fields(FusionConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Claves desconocidas en [fusion]: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    return FusionConfig(**data)
