"""Fuentes de frames propias del ejecutor: video grabado (perfil ``replay``).

El pipeline de #30 solo reconoce RTSP/USB. Para reproducir un video grabado se
inyecta una ``SourceFactory`` que ignora la URI placeholder y abre el archivo.
El frame sigue viviendo solo en el plano de video.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pipeline import Frame, SourceError, StreamSource
from pipeline.fake import FakeSourceFactory

PLACEHOLDER_URI = "usb:0"
"""URI válida que exige ``StreamSource``; la fuente de archivo no la usa."""


class FileFrameSource:
    """Lee un video con cv2 a ``fps`` y vuelve al inicio al terminar (bucle)."""

    def __init__(self, path: str, fps: float) -> None:
        self._path = path
        self._period = 1.0 / fps
        self._capture: Any = None
        self._next_at = 0.0

    def open(self) -> None:
        if not Path(self._path).is_file():
            raise SourceError(f"Video de replay no encontrado: {self._path}")
        import cv2  # importación diferida

        capture = cv2.VideoCapture(self._path)
        if not capture.isOpened():
            capture.release()
            raise SourceError(f"No se pudo abrir el video: {self._path}")
        self._capture = capture
        self._next_at = time.monotonic()

    def read(self) -> Frame:
        if self._capture is None:
            raise SourceError("La fuente de archivo no está abierta")
        delay = self._next_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_at = (
            max(self._next_at, time.monotonic() - self._period) + self._period
        )
        ok, image = self._capture.read()
        if not ok or image is None:
            self._capture.set(1, 0)  # cv2.CAP_PROP_POS_FRAMES = 1: rebobinar
            ok, image = self._capture.read()
            if not ok or image is None:
                raise SourceError(f"Video vacío o ilegible: {self._path}")
        height, width = image.shape[:2]
        return Frame(width=width, height=height, data=image)

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()


class FileSourceFactory:
    """``SourceFactory`` que abre siempre el mismo archivo de video."""

    def __init__(self, path: str, fps: float) -> None:
        self._path = path
        self._fps = fps

    def __call__(self, source: StreamSource) -> FileFrameSource:
        return FileFrameSource(self._path, self._fps)


def fake_source_factory(stream_id: str, fps: float) -> FakeSourceFactory:
    """Cámara sintética de 640x480 a ``fps`` (el frame es un buffer mínimo)."""
    factory = FakeSourceFactory(default_interval=1.0 / fps)
    camera = factory.camera(stream_id)
    camera.width, camera.height = 640, 480
    return factory
