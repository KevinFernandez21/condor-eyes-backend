"""Contrato del pipeline de video compartido por todas las cámaras."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class SourceKind(StrEnum):
    """Tipo de cámara soportado por el pipeline."""

    RTSP = "rtsp"
    USB = "usb"


_RTSP_SCHEMES = ("rtsp://", "rtsps://")
_USB_PREFIXES = ("usb://", "usb:", "v4l2://")


def _classify_uri(uri: str) -> SourceKind:
    """Clasifica la URI y valida su forma; lanza ValueError en español."""
    value = uri.strip()
    for scheme in _RTSP_SCHEMES:
        if value.startswith(scheme):
            if len(value) == len(scheme):
                raise ValueError(f"URI RTSP sin host: {uri!r}")
            return SourceKind.RTSP
    if value.startswith("/dev/video") and value[len("/dev/video") :].isdigit():
        return SourceKind.USB
    for prefix in _USB_PREFIXES:
        if value.startswith(prefix):
            rest = value[len(prefix) :]
            if rest.isdigit() or (
                rest.startswith("/dev/video") and rest[len("/dev/video") :].isdigit()
            ):
                return SourceKind.USB
            raise ValueError(
                f"URI USB inválida (se espera índice o /dev/videoN): {uri!r}"
            )
    raise ValueError(
        f"URI no soportada: {uri!r}. Use rtsp://, rtsps://, usb:N o /dev/videoN"
    )


@dataclass(frozen=True, slots=True)
class StreamSource:
    """Configuración mínima de una fuente RTSP o USB."""

    stream_id: str
    uri: str

    def __post_init__(self) -> None:
        if not self.stream_id.strip():
            raise ValueError("El stream_id no puede estar vacío")
        _classify_uri(self.uri)

    @property
    def kind(self) -> SourceKind:
        """Tipo de fuente deducido de la URI."""
        return _classify_uri(self.uri)


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
