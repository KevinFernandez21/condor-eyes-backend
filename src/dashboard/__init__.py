"""Dashboard del sistema multiagente (#43): cliente de la API de observabilidad (#42)."""

from .config import DashboardConfig
from .server import DashboardConfigurationError, create_app

__all__ = ["DashboardConfig", "DashboardConfigurationError", "create_app"]
