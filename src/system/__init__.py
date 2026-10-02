"""Ejecutor local del sistema multiagente completo (issue #41).

API estable para otros componentes (observabilidad, dashboard):
``SystemApp``, ``SystemApp.runtime``, ``SystemApp.hub``, ``SystemApp.snapshot()``.
"""

from .app import SystemApp
from .config import ConfigError, SystemConfig, load_system_config
from .plugins import KNOWN_PLUGINS, PluginRegistry
from .stats import InstrumentedHub

__all__ = [
    "KNOWN_PLUGINS",
    "ConfigError",
    "InstrumentedHub",
    "PluginRegistry",
    "SystemApp",
    "SystemConfig",
    "load_system_config",
]
