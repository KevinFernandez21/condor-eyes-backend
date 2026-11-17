"""Registro de plugins opcionales descubiertos por importación.

Location, identity, actuation y tagbridge viven en PRs aparte. El ejecutor no
los importa de forma dura: pregunta con ``importlib.util.find_spec`` si el
módulo existe y, si no, lo reporta con estado ``not_installed`` mientras el
resto del sistema sigue funcionando (con simuladores integrados).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ModuleFinder = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class PluginSpec:
    """Plugin opcional: nombre lógico, módulo a detectar y el issue que lo trae."""

    name: str
    module: str
    issue: str
    purpose: str


KNOWN_PLUGINS: dict[str, PluginSpec] = {
    spec.name: spec
    for spec in (
        PluginSpec("location", "location", "#32", "Localización de tags por zona"),
        PluginSpec("tagbridge", "tagbridge", "#37", "Puente BLE del tag XIAO ESP32-C6"),
        PluginSpec("actuation", "actuation", "#33", "Actuación pan-tilt"),
        PluginSpec(
            "identity", "faceid.identity", "#36", "Identidad facial por búsqueda"
        ),
    )
}


@dataclass(frozen=True, slots=True)
class PluginInfo:
    name: str
    module: str
    installed: bool


def module_available(module: str) -> bool:
    """``find_spec`` sin lanzar: un padre ausente o roto cuenta como no instalado."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


class PluginRegistry:
    """Descubre qué plugins están presentes y describe su estado."""

    def __init__(
        self,
        specs: dict[str, PluginSpec] | None = None,
        *,
        finder: ModuleFinder = module_available,
    ) -> None:
        self._specs = dict(specs or KNOWN_PLUGINS)
        self._finder = finder

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def _installed(self, spec: PluginSpec) -> bool:
        try:
            return bool(self._finder(spec.module))
        except (ImportError, ValueError, AttributeError):
            return False

    def discover(self) -> dict[str, PluginInfo]:
        return {
            name: PluginInfo(name, spec.module, self._installed(spec))
            for name, spec in self._specs.items()
        }

    def is_installed(self, name: str) -> bool:
        return self._installed(self._specs[name])

    def health(self, name: str) -> dict[str, Any]:
        """Estado del plugin como metadata JSON: ``installed`` o ``not_installed``."""
        spec = self._specs[name]
        if self._installed(spec):
            return {
                "status": "installed",
                "detail": f"módulo '{spec.module}' disponible",
            }
        return {
            "status": "not_installed",
            "detail": (
                f"módulo '{spec.module}' no encontrado; {spec.purpose} llega con "
                f"{spec.issue}. Se usa el simulador integrado."
            ),
        }
