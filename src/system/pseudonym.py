"""Seudónimos de personas y tags de la simulación (HMAC-SHA256 con clave secreta).

Ningún identificador en claro (``person-sim-01``) llega al bus: se publica
``p-xxxx`` / ``tag-xxxxxxxx``, coherente con el resto de módulos. La clave es
aleatoria por ejecución, o derivada de la semilla para que una corrida
reproducible produzca los mismos seudónimos.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Iterable


class Pseudonymizer:
    """Mapea un identificador a un seudónimo estable para una misma clave."""

    def __init__(self, key: bytes) -> None:
        if not key:
            raise ValueError("La clave del seudonimizador no puede estar vacía")
        self._key = key

    @classmethod
    def random(cls) -> Pseudonymizer:
        return cls(os.urandom(32))

    @classmethod
    def from_seed(cls, seed: int) -> Pseudonymizer:
        return cls(hashlib.sha256(f"condor-sim-seed:{seed}".encode()).digest())

    def _digest(self, scope: str, value: str) -> str:
        mac = hmac.new(self._key, f"{scope}:{value}".encode(), hashlib.sha256)
        return mac.hexdigest()

    def person(self, handle: str) -> str:
        return f"p-{self._digest('person', handle)[:4]}"

    def tag(self, handle: str) -> str:
        return f"tag-{self._digest('tag', handle)[:8]}"

    def check_unique(self, handles: Iterable[str]) -> None:
        """Falla si dos identificadores distintos comparten seudónimo de persona."""
        seen: dict[str, str] = {}
        for handle in handles:
            alias = self.person(handle)
            if alias in seen and seen[alias] != handle:
                raise ValueError(
                    f"colisión de seudónimos: {handle!r} y {seen[alias]!r} -> {alias}"
                )
            seen[alias] = handle
