"""Ejecutor local del sistema multiagente completo (issue #41).

API estable para otros componentes (observabilidad, dashboard):
``SystemApp``, ``SystemApp.runtime``, ``SystemApp.hub``, ``SystemApp.snapshot()``.
"""

from .app import SystemApp
from .config import ConfigError, SystemConfig, load_system_config
from .plugins import KNOWN_PLUGINS, PluginRegistry

__all__ = [
    "KNOWN_PLUGINS",
    "ConfigError",
    "PluginRegistry",
    "SystemApp",
    "SystemConfig",
    "load_system_config",
]
