"""Configuración del dashboard (variables de entorno ``CONDOR_*``)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlencode

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_SITE_MAP = "configs/site_map.toml"


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Dónde está la API de observabilidad y cómo servir el dashboard."""

    api_url: str = DEFAULT_API_URL
    token: str | None = None
    site_map_path: str | None = DEFAULT_SITE_MAP
    poll_interval_s: float = 1.0
    feed_size: int = 300

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DashboardConfig:
        env = os.environ if env is None else env
        token = (env.get("CONDOR_API_TOKEN") or "").strip() or None
        return cls(
            api_url=(env.get("CONDOR_API_URL") or DEFAULT_API_URL).strip().rstrip("/"),
            token=token,
            site_map_path=(env.get("CONDOR_SITE_MAP") or DEFAULT_SITE_MAP).strip() or None,
        )

    def ws_url(self) -> str:
        """URL del WebSocket; el token va por query porque el navegador/cliente
        WS de la API no admite cabeceras (ver docs/observability-api.md)."""
        base = self.api_url.rstrip("/")
        if base.startswith("https://"):
            base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://") :]
        url = f"{base}/ws"
        if self.token:
            url += "?" + urlencode({"token": self.token})
        return url

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}
