"""Fuentes de frames del plano de video (el frame nunca sale de este plano)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .shared_pipeline import SourceKind, StreamSource


class SourceError(RuntimeError):
    """Fallo de apertura o lectura de una fuente (desconexión, fin de stream)."""


@dataclass(frozen=True, slots=True)
class Frame:
    """Frame decodificado; vive solo dentro del pipeline, jamás en el bus."""

    width: int
    height: int
    data: Any = None


class FrameSource(Protocol):
    """Origen de frames de una cámara; implementaciones: OpenCV, falsa, etc."""

    def open(self) -> None:
        """Abre la cámara; lanza excepción si no puede."""

    def read(self) -> Frame:
        """Bloquea hasta el siguiente frame; lanza excepción si se pierde."""

    def close(self) -> None:
        """Libera la cámara; debe ser idempotente."""


SourceFactory = Callable[[StreamSource], FrameSource]


class OpenCVFrameSource:
    """Backend CPU basado en cv2.VideoCapture (RTSP vía FFmpeg, USB vía V4L2).

    cv2 se importa de forma diferida para que el módulo cargue en cualquier
    máquina.
    """

    def __init__(self, source: StreamSource) -> None:
        self._source = source
        self._capture: Any = None

    @staticmethod
    def _target(source: StreamSource) -> str | int:
        uri = source.uri.strip()
        if source.kind is SourceKind.RTSP:
            return uri
        for prefix in ("usb://", "usb:", "v4l2://"):
            if uri.startswith(prefix):
                uri = uri[len(prefix) :]
                break
        return int(uri) if uri.isdigit() else uri

    def open(self) -> None:
        import cv2  # importación diferida

        capture = cv2.VideoCapture(self._target(self._source))
        if not capture.isOpened():
            capture.release()
            raise SourceError(f"No se pudo abrir la fuente {self._source.uri!r}")
        self._capture = capture

    def read(self) -> Frame:
        if self._capture is None:
            raise SourceError("La fuente no está abierta")
        ok, image = self._capture.read()
        if not ok or image is None:
            raise SourceError(f"Sin frames de {self._source.uri!r} (stream cortado)")
        height, width = image.shape[:2]
        return Frame(width=width, height=height, data=image)

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()


def opencv_source_factory(source: StreamSource) -> FrameSource:
    """Fábrica por defecto para ejecutar en CPU (laptop de desarrollo)."""
    return OpenCVFrameSource(source)
