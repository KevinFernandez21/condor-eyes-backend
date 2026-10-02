"""Fuentes falsas, seguras para CPU, para probar el pipeline sin cámaras."""

from __future__ import annotations

import threading
import time

from .shared_pipeline import StreamSource
from .sources import Frame, FrameSource, SourceError

_STALL_CEILING = (
    10.0  # un read() bloqueado nunca dura más que esto (evita hilos colgados)
)


class FakeCamera:
    """Controlador de una cámara simulada: permite provocar fallos a demanda."""

    def __init__(self, interval: float = 0.01, width: int = 64, height: int = 48):
        self.interval = interval
        self.width = width
        self.height = height
        self.open_attempts = 0
        self.open_sources = 0
        self._lock = threading.Lock()
        self._open_failures = 0
        self._connected = True
        self._release = threading.Event()
        self._release.set()

    def fail_opens(self, count: int) -> None:
        """Hace fallar las próximas `count` aperturas."""
        with self._lock:
            self._open_failures = count

    def disconnect(self) -> None:
        """Simula caída de red: lecturas y aperturas fallan."""
        self._connected = False

    def reconnect(self) -> None:
        self._connected = True

    def stall(self) -> None:
        """Simula cámara congelada: read() se bloquea sin devolver frames."""
        self._release.clear()

    def resume(self) -> None:
        self._release.set()

    def _try_open(self) -> None:
        with self._lock:
            self.open_attempts += 1
            if self._open_failures > 0:
                self._open_failures -= 1
                raise ConnectionError("apertura simulada fallida")
        if not self._connected:
            raise ConnectionError("cámara simulada desconectada")
        with self._lock:
            self.open_sources += 1

    def _closed(self) -> None:
        with self._lock:
            self.open_sources -= 1


class FakeFrameSource:
    """FrameSource que obedece a un FakeCamera."""

    def __init__(self, camera: FakeCamera) -> None:
        self._camera = camera
        self._opened = False
        self._counter = 0

    def open(self) -> None:
        self._camera._try_open()
        self._opened = True

    def read(self) -> Frame:
        camera = self._camera
        if not self._opened:
            raise SourceError("La fuente falsa no está abierta")
        if not camera._release.is_set():
            camera._release.wait(timeout=_STALL_CEILING)
        if not camera._connected:
            raise ConnectionError("cámara simulada desconectada")
        time.sleep(camera.interval)
        self._counter += 1
        # Datos reales (bytes) para poder comprobar que nunca llegan al bus.
        return Frame(camera.width, camera.height, data=bytes(8))

    def close(self) -> None:
        if self._opened:
            self._opened = False
            self._camera._closed()


class FakeSourceFactory:
    """Fábrica de fuentes falsas; crea un FakeCamera por stream_id."""

    def __init__(self, default_interval: float = 0.01) -> None:
        self._default_interval = default_interval
        self._cameras: dict[str, FakeCamera] = {}
        self._lock = threading.Lock()

    def camera(self, stream_id: str) -> FakeCamera:
        with self._lock:
            if stream_id not in self._cameras:
                self._cameras[stream_id] = FakeCamera(self._default_interval)
            return self._cameras[stream_id]

    def cameras(self) -> list[FakeCamera]:
        with self._lock:
            return list(self._cameras.values())

    def __call__(self, source: StreamSource) -> FrameSource:
        return FakeFrameSource(self.camera(source.stream_id))
