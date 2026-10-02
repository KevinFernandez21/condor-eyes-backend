"""Pipeline de video en vivo: ciclo de vida, reconexión, watchdog y backpressure.

La lógica es Python puro e independiente del backend de captura. El backend
se inyecta mediante una `SourceFactory` (OpenCV en CPU, fuente falsa en
tests; el grafo DeepStream/GStreamer objetivo se documenta en
`docs/video-pipeline.md`).

Regla central: los frames viven solo en este plano. Hacia los agentes salen
únicamente `FrameMetadata` y `StreamHealth`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .buffer import DropOldestQueue
from .health import BackoffPolicy, StreamHealth, StreamState
from .metadata import FrameMetadata
from .shared_pipeline import StreamSource
from .sources import Frame, SourceFactory, opencv_source_factory

logger = logging.getLogger(__name__)

Processor = Callable[[str, Frame], "list[Mapping[str, Any]] | None"]
MetadataCallback = Callable[[FrameMetadata], None]
StatusCallback = Callable[[StreamHealth], None]


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Parámetros operativos del pipeline."""

    backoff: BackoffPolicy = field(default_factory=BackoffPolicy)
    watchdog_timeout: float = 5.0
    watchdog_interval: float = 1.0
    max_buffered_frames: int = 4
    max_retries: int | None = None
    max_streams: int = 8
    join_timeout: float = 3.0

    def __post_init__(self) -> None:
        if self.watchdog_timeout <= 0 or self.watchdog_interval <= 0:
            raise ValueError("Los tiempos del watchdog deben ser positivos")
        if self.max_buffered_frames < 1:
            raise ValueError("max_buffered_frames debe ser >= 1")
        if self.max_retries is not None and self.max_retries < 1:
            raise ValueError("max_retries debe ser >= 1 o None")
        if not 1 <= self.max_streams <= 8:
            raise ValueError("max_streams debe estar entre 1 y 8 (Orin Nano)")


@dataclass(slots=True)
class _Stream:
    source: StreamSource
    queue: DropOldestQueue[Frame]
    state: StreamState = StreamState.IDLE
    last_frame_at: datetime | None = None
    last_progress: float = 0.0  # reloj monotónico, base del watchdog
    watching: bool = False  # True solo mientras se abre/lee (no en backoff)
    retry_count: int = 0
    last_error: str | None = None
    total_reconnects: int = 0
    frames_received: int = 0
    frames_dropped: int = 0
    callback_errors: int = 0
    next_index: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None


class LiveVideoPipeline:
    """Implementa `SharedVideoPipeline` con hilos y fuentes inyectables."""

    def __init__(
        self,
        source_factory: SourceFactory = opencv_source_factory,
        *,
        config: PipelineConfig | None = None,
        on_metadata: MetadataCallback | None = None,
        on_status: StatusCallback | None = None,
        processor: Processor | None = None,
    ) -> None:
        self._factory = source_factory
        self._config = config or PipelineConfig()
        self._on_metadata = on_metadata
        self._on_status = on_status
        self._processor = processor
        self._lock = threading.RLock()
        self._streams: dict[str, _Stream] = {}
        self._running = False
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._dispatcher: threading.Thread | None = None

    # ------------------------------------------------------------------ API

    def add_source(self, source: StreamSource) -> None:
        with self._lock:
            if source.stream_id in self._streams:
                raise ValueError(
                    f"El stream_id {source.stream_id!r} ya está registrado"
                )
            if len(self._streams) >= self._config.max_streams:
                raise ValueError(
                    f"Se alcanzó el máximo de {self._config.max_streams} streams"
                )
            stream = _Stream(source, DropOldestQueue(self._config.max_buffered_frames))
            self._streams[source.stream_id] = stream
            if self._running:
                self._launch(stream)

    def remove_source(self, stream_id: str) -> None:
        with self._lock:
            stream = self._get(stream_id)
            stream.cancel.set()
            worker = stream.worker
            del self._streams[stream_id]
            stream.queue.clear()
        self._join(worker)
        self._notify(self._snapshot_of(stream, StreamState.STOPPED))

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop.clear()
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop, name="video-dispatch", daemon=True
            )
            self._watchdog = threading.Thread(
                target=self._watchdog_loop, name="video-watchdog", daemon=True
            )
            self._dispatcher.start()
            self._watchdog.start()
            for stream in self._streams.values():
                self._launch(stream)

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop.set()
            streams = list(self._streams.values())
            for stream in streams:
                stream.cancel.set()
            threads = [s.worker for s in streams] + [self._watchdog, self._dispatcher]
        self._wakeup.set()
        deadline = time.monotonic() + self._config.join_timeout
        for thread in threads:
            self._join(thread, deadline)
        stopped: list[StreamHealth] = []
        with self._lock:
            for stream in streams:
                stream.worker = None
                stream.queue.clear()
                stream.state = StreamState.STOPPED
                stopped.append(self._snapshot(stream))
        for snapshot in stopped:
            self._notify(snapshot)

    def health(self) -> Mapping[str, object]:
        with self._lock:
            return {
                "running": self._running,
                "streams": {
                    sid: self._snapshot(stream).to_payload()
                    for sid, stream in self._streams.items()
                },
            }

    def stream_health(self, stream_id: str) -> StreamHealth:
        with self._lock:
            return self._snapshot(self._get(stream_id))

    # ------------------------------------------------------------ internals

    def _get(self, stream_id: str) -> _Stream:
        try:
            return self._streams[stream_id]
        except KeyError:
            raise KeyError(f"Stream desconocido: {stream_id!r}") from None

    def _snapshot(self, stream: _Stream) -> StreamHealth:
        return self._snapshot_of(stream, stream.state)

    @staticmethod
    def _snapshot_of(stream: _Stream, state: StreamState) -> StreamHealth:
        return StreamHealth(
            stream_id=stream.source.stream_id,
            state=state,
            last_frame_at=stream.last_frame_at,
            retry_count=stream.retry_count,
            last_error=stream.last_error,
            total_reconnects=stream.total_reconnects,
            frames_received=stream.frames_received,
            frames_dropped=stream.frames_dropped,
            queue_depth=len(stream.queue),
            callback_errors=stream.callback_errors,
        )

    def _notify(self, health: StreamHealth) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(health)
        except Exception:
            logger.exception("Fallo en on_status de %s", health.stream_id)

    def _set_state(
        self, stream: _Stream, cancel: threading.Event, state: StreamState
    ) -> bool:
        """Cambia el estado si el worker sigue vigente; notifica fuera del lock."""
        with self._lock:
            if cancel.is_set():
                return False
            stream.state = state
            snapshot = self._snapshot(stream)
        self._notify(snapshot)
        return True

    @staticmethod
    def _join(thread: threading.Thread | None, deadline: float | None = None) -> None:
        if thread is None or thread is threading.current_thread():
            return
        remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        thread.join(timeout=remaining if remaining is not None else 2.0)

    def _launch(self, stream: _Stream, initial_delay: float = 0.0) -> None:
        """Arranca un worker con un token de cancelación nuevo (bajo lock)."""
        cancel = threading.Event()
        stream.cancel = cancel
        stream.watching = False
        stream.state = StreamState.CONNECTING
        stream.worker = threading.Thread(
            target=self._worker_loop,
            args=(stream, cancel, initial_delay),
            name=f"video-{stream.source.stream_id}",
            daemon=True,
        )
        stream.worker.start()

    def _worker_loop(
        self, stream: _Stream, cancel: threading.Event, initial_delay: float
    ) -> None:
        policy = self._config.backoff
        if initial_delay and cancel.wait(initial_delay):
            return
        while not cancel.is_set():
            if not self._set_state(stream, cancel, self._attempt_state(stream)):
                return
            source = None
            try:
                source = self._factory(stream.source)
                with self._lock:
                    stream.last_progress = time.monotonic()
                    stream.watching = True
                source.open()
                if not self._set_state(stream, cancel, StreamState.CONNECTED):
                    return
                while not cancel.is_set():
                    frame = source.read()
                    if cancel.is_set():
                        return
                    self._on_frame(stream, cancel, frame)
            except Exception as exc:  # noqa: BLE001 - cualquier fallo de fuente reconecta
                if not cancel.is_set():
                    self._record_error(stream, cancel, f"{type(exc).__name__}: {exc}")
            finally:
                self._safe_close(source)
            if cancel.is_set():
                return
            with self._lock:
                if cancel.is_set():
                    return
                stream.retry_count += 1
                stream.total_reconnects += 1
                attempt = stream.retry_count
                give_up = (
                    self._config.max_retries is not None
                    and attempt > self._config.max_retries
                )
            if give_up:
                self._set_state(stream, cancel, StreamState.FAILED)
                return
            with self._lock:
                stream.watching = False
            self._set_state(stream, cancel, StreamState.RECONNECTING)
            if cancel.wait(policy.delay(attempt - 1)):
                return

    @staticmethod
    def _attempt_state(stream: _Stream) -> StreamState:
        return (
            StreamState.RECONNECTING if stream.retry_count else StreamState.CONNECTING
        )

    @staticmethod
    def _safe_close(source: Any) -> None:
        if source is None:
            return
        try:
            source.close()
        except Exception:
            logger.exception("Error al cerrar la fuente")

    def _record_error(
        self, stream: _Stream, cancel: threading.Event, message: str
    ) -> None:
        with self._lock:
            if not cancel.is_set():
                stream.last_error = message

    def _on_frame(self, stream: _Stream, cancel: threading.Event, frame: Frame) -> None:
        recovered: StreamHealth | None = None
        with self._lock:
            if cancel.is_set():
                return
            stream.last_progress = time.monotonic()
            stream.last_frame_at = datetime.now(UTC)
            stream.frames_received += 1
            if stream.retry_count:
                stream.retry_count = 0
                recovered = self._snapshot(stream)
            stream.frames_dropped += stream.queue.put(frame)
        if recovered is not None:
            self._notify(recovered)
        self._wakeup.set()

    # ------------------------------------------------------------ dispatcher

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            self._wakeup.wait(timeout=0.1)
            self._wakeup.clear()
            while not self._stop.is_set() and self._dispatch_round():
                pass

    def _dispatch_round(self) -> bool:
        """Entrega un frame por stream (round-robin); True si hubo trabajo."""
        with self._lock:
            streams = list(self._streams.values())
        worked = False
        for stream in streams:
            frame = stream.queue.get_nowait()
            if frame is None:
                continue
            worked = True
            self._deliver(stream, frame)
        return worked

    def _deliver(self, stream: _Stream, frame: Frame) -> None:
        stream_id = stream.source.stream_id
        try:
            detections = (
                self._processor(stream_id, frame) if self._processor else None
            ) or []
            with self._lock:
                index = stream.next_index
                stream.next_index += 1
            meta = FrameMetadata(
                stream_id=stream_id,
                frame_index=index,
                timestamp=datetime.now(UTC),
                width=frame.width,
                height=frame.height,
                detections=tuple(detections),
            )
            if self._on_metadata is not None:
                self._on_metadata(meta)
        except Exception:
            logger.exception("Fallo al procesar/entregar un frame de %s", stream_id)
            with self._lock:
                stream.callback_errors += 1

    # --------------------------------------------------------------- watchdog

    def _watchdog_loop(self) -> None:
        while not self._stop.wait(self._config.watchdog_interval):
            self._check_stalls()

    def _check_stalls(self) -> None:
        notifications: list[StreamHealth] = []
        with self._lock:
            if not self._running:
                return
            now = time.monotonic()
            for stream in self._streams.values():
                if not stream.watching:
                    continue
                silent = now - stream.last_progress
                if silent <= self._config.watchdog_timeout:
                    continue
                stream.watching = False
                stream.cancel.set()  # el worker colgado queda huérfano y se retira solo
                stream.last_error = f"watchdog: sin frames durante {silent:.1f}s"
                stream.retry_count += 1
                stream.total_reconnects += 1
                give_up = (
                    self._config.max_retries is not None
                    and stream.retry_count > self._config.max_retries
                )
                if give_up:
                    stream.state = StreamState.FAILED
                else:
                    delay = self._config.backoff.delay(stream.retry_count - 1)
                    self._launch(stream, initial_delay=delay)
                    stream.state = StreamState.RECONNECTING
                notifications.append(self._snapshot(stream))
        for snapshot in notifications:
            self._notify(snapshot)
