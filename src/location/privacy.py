"""Seudonimización de identificadores para logs y mensajes del bus.

Los IDs de tag y de persona nunca deben aparecer en claro fuera del repositorio
de personal. Se usa HMAC-SHA256 con clave secreta: sin la clave no se puede
revertir ni correlacionar entre despliegues.
"""

from __future__ import annotations

import hashlib
import hmac
import os


class Pseudonymizer:
    """Genera referencias opacas y estables para IDs sensibles."""

    def __init__(self, key: bytes) -> None:
        if not key:
            raise ValueError("la clave de seudonimización no puede estar vacía")
        self._key = key

    @classmethod
    def random(cls) -> Pseudonymizer:
        """Clave efímera: las referencias cambian en cada arranque del proceso."""
        return cls(os.urandom(32))

    def ref(self, kind: str, value: str) -> str:
        """Referencia `<kind>-<12 hex>`; `kind` separa dominios (tag, person)."""
        digest = hmac.new(
            self._key, f"{kind}:{value}".encode(), hashlib.sha256
        ).hexdigest()
        return f"{kind}-{digest[:12]}"
