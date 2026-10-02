# Pipeline de video compartido (plano de video)

Implementa el issue #13. El plano de video decodifica cada cámara una sola
vez, aplica backpressure acotado y entrega a los agentes **solo metadata**
(`FrameMetadata`, `StreamHealth`). Ningún frame ni buffer codificado cruza
el bus.

## Estado de validación

| Parte | Validación |
|---|---|
| Ciclo de vida, reconexión, backoff, watchdog, buffers acotados | Probado con fuentes falsas (`pytest`) |
| Backend CPU (`OpenCVFrameSource`) | Probado con `cv2` simulado; **no** probado con una cámara real |
| Grafo `cpu_graph` (GStreamer CPU) | Solo generación de texto; no ejecutado |
| Grafo `deepstream_graph` (NVDEC/nvinfer/nvtracker) | **No validado**: no hay Jetson. Es la especificación a verificar en hardware |

## Módulos (`src/pipeline/`)

- `shared_pipeline.py`: `StreamSource` (valida RTSP/USB) y el `Protocol` `SharedVideoPipeline`.
- `live_pipeline.py`: `LiveVideoPipeline`, `PipelineConfig`. Lógica Python pura, independiente del backend.
- `health.py`: `StreamState`, `StreamHealth`, `BackoffPolicy`.
- `buffer.py`: `DropOldestQueue` (cola acotada que descarta lo más viejo).
- `metadata.py`: `FrameMetadata` y `ensure_metadata_only` (rechaza arrays/bytes).
- `sources.py`: `FrameSource`, `OpenCVFrameSource` (cv2 diferido).
- `fake.py`: `FakeCamera`/`FakeSourceFactory` para tests sin hardware.
- `gst_graph.py`: genera las descripciones `gst-launch` (CPU y DeepStream).

## Modelo operativo

Un hilo lector por stream, un hilo despachador compartido y un watchdog.

```
 lector cam-1 ─┐                       ┌─ processor(stream_id, frame) -> detecciones
 lector cam-2 ─┼─► DropOldestQueue ──► despachador (round-robin) ─► on_metadata(FrameMetadata)
 lector cam-N ─┘   (por stream, N=4)   └─ nunca bloquea al lector      │
                                                                       ▼
 watchdog ──► reinicia lectores sin frames               (agentes: solo metadata)
```

- **Estados**: `idle -> connecting -> connected -> reconnecting -> failed | stopped`.
- **Reconexión**: ante cualquier error de apertura o lectura el stream pasa a
  `reconnecting` y reintenta con backoff exponencial (`1s, 2s, 4s ... 30s`,
  jitter opcional). Los demás streams no se ven afectados. `retry_count` se
  reinicia con el primer frame recibido; `total_reconnects` es acumulado.
  Con `max_retries` el stream termina en `failed`.
- **Watchdog**: si un stream abre/lee sin progreso durante `watchdog_timeout`
  se invalida su lector (aunque esté bloqueado en `read()`) y se lanza uno nuevo.
- **Backpressure**: cada stream tiene una cola de `max_buffered_frames` que
  descarta el frame más viejo; el lector nunca se bloquea. `frames_dropped` y
  `queue_depth` exponen el descarte.
- **Salud por stream** (`stream_health` / `health()`): `state`,
  `last_frame_at`, `retry_count`, `last_error`, `total_reconnects`,
  `frames_received`, `frames_dropped`, `queue_depth`, `callback_errors`.
- **Apagado**: `stop()` cancela todos los hilos (incluidos los dormidos en
  backoff), los une con tiempo límite y deja los streams en `stopped`.
  Es idempotente y el pipeline puede reiniciarse.

## Uso en laptop (CPU)

```python
from pipeline import LiveVideoPipeline, StreamSource

pipe = LiveVideoPipeline(on_metadata=print)  # OpenCV por defecto
pipe.add_source(StreamSource("cam-01", "rtsp://192.168.1.10:554/stream1"))
pipe.add_source(StreamSource("cam-usb", "usb:0"))
pipe.start()
print(pipe.health())
pipe.stop()
```

Para tests usar `FakeSourceFactory` (`pipeline.fake`), que permite provocar
`disconnect()`, `stall()` y `fail_opens(n)` por cámara.

## Grafo objetivo en Jetson Orin Nano (no validado)

```
rtspsrc ! rtph264depay ! h264parse ! nvv4l2decoder ──┐   (NVDEC, uno por cámara)
v4l2src ! videoconvert ! nvvideoconvert (NVMM) ──────┤
                                                     ▼
                         nvstreammux (batch N, live-source=1)
                                                     ▼
                         nvinfer (engine TensorRT FP16 único)
                                                     ▼
                         nvtracker (NvDCF o ByteTrack, ARCH-003 abierta)
                                                     ▼
                         appsink (drop=true, max-buffers=2) ──► metadata
```

`pipeline.gst_graph.deepstream_graph(sources, engine_config=...)` genera la
cadena `gst-launch` equivalente (batch 1-8). El `appsink` solo extrae
metadata (`NvDsBatchMeta`) hacia `FrameMetadata`; los buffers NVMM no salen
del plano de video. Un único engine FP16 compartido, nunca uno por cámara.

Pendiente en hardware: validar el grafo, elegir tracker (ARCH-003), medir
latencia/NVDEC con 4-8 streams y escribir el backend `FrameSource`/probe
basado en GStreamer (`gi`, importado de forma diferida).
