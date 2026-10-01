"""Contrato del pipeline GStreamer compartido por todas las cámaras."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StreamSource:
    """Configuración mínima de una fuente RTSP o USB."""

    stream_id: str
    uri: str


class SharedVideoPipeline(Protocol):
    """Frontera del plano de video; sus buffers nunca pasan al bus."""

    def add_source(self, source: StreamSource) -> None:
        """Añade una cámara al batch compartido."""

    def remove_source(self, stream_id: str) -> None:
        """Retira una cámara del batch compartido."""

    def start(self) -> None:
        """Inicia el pipeline GStreamer."""

    def stop(self) -> None:
        """Detiene el pipeline y libera recursos."""

    def health(self) -> Mapping[str, object]:
        """Devuelve únicamente metadata operativa del pipeline."""
