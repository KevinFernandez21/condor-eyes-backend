"""Saneado de URIs: las credenciales RTSP no deben salir del plano de video."""

from __future__ import annotations

import re

# scheme://usuario[:clave]@  (cualquier esquema con ://)
_USERINFO = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s@]+@")


def redact_credentials(text: str) -> str:
    """Elimina `usuario:clave@` de cualquier URI contenida en `text`."""
    return _USERINFO.sub(lambda match: match.group("scheme"), text)


def sanitize_uri(uri: str) -> str:
    """Devuelve la URI sin credenciales, conservando esquema/host/puerto/ruta."""
    return redact_credentials(uri)
