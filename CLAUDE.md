# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Estado del repo

Proyecto en Fase 0: videovigilancia multiagente para **Jetson Orin Nano 8GB**, todo en **Python + AgentScope**, objetivo 4-8 streams 1080p con inferencia compartida. Reemplaza al repo anterior `condor-eye-backend` (ISR para drones); no se reutiliza nada de él.

Lo único implementado es `src/compare/`, un harness para elegir el modelo base (decisión ARCH-002: YOLOv8n vs PeopleNet). Los módulos `src/agents/`, `src/pipeline/` y `src/bus/` descritos en el README y en `docs/architecture.md` **todavía no existen**; son el diseño a seguir al crearlos.

Todavía no hay `pyproject.toml`, `requirements.txt`, tests, Dockerfile ni configuración de pre-commit. El stack de calidad aprobado (`docs/tech-stack.md`) es `ruff`, `mypy` y `pytest`. Al añadir tests, `FakeDetector` permite probar el harness sin GPU ni pesos.

La documentación y los mensajes de error están en español; mantén ese idioma.

## Harness de comparación (`src/compare/`)

Está pensado para llamarse desde un notebook del usuario, no como CLI. Al no haber paquete instalable, hay que añadir `src` al `sys.path`:

```python
import sys; sys.path.insert(0, "src")
from compare import YOLOv8nDetector, PeopleNetDetector, FakeDetector, compare_models
rows = compare_models([FakeDetector(), YOLOv8nDetector()], source="video.mp4", frames=100)
```

- Todos los detectores cumplen el `Protocol` `Detector` (`name`, `warmup`, `infer(frame) -> list[dict]`, `close`). Un detector nuevo debe respetarlo para entrar en `compare_models`, que llama a `close()` aunque falle.
- Las dependencias pesadas (`ultralytics`, `tensorrt`, `onnxruntime`, `cv2`) se importan de forma diferida dentro de los métodos, para que el módulo cargue en cualquier máquina. Mantén ese patrón.
- `source` en `benchmark_detector` admite un callable, una lista de frames, una ruta de video (cv2) o `None`.
- `PeopleNetDetector` con `.engine` solo funciona en Jetson. La ruta `.onnx` corre en CPU, pero devuelve solo las formas de salida (sin postproceso), y la inferencia TensorRT directa lanza `NotImplementedError`. Sirve para medir latencia, no para comparar detecciones.

## Arquitectura objetivo (ver `docs/architecture.md`)

Dos planos separados. Esta separación es la regla central del diseño:

- **Plano video**: un único pipeline GStreamer compartido (NVDEC → `nvstreammux` batch N → `nvinfer` FP16 → `nvtracker` → appsink) en `src/pipeline/`, fuera de AgentScope.
- **Plano agentes**: agentes AgentScope que intercambian solo metadata (detecciones, tracks, eventos) por MsgHub: `ingest` (uno por cámara), `inference` (único, escala por batch), `tracker`, `event` (ReActAgent), `storage` (clips + SQLite WAL), `supervisor` (ReActAgent) y `comms` (FastAPI HTTP/WS).

Reglas de `docs/tech-stack.md` que no se deben romper:
- Ningún frame sale del pipeline hacia el MsgHub.
- Un solo engine TensorRT FP16 versionado y compartido por todos los streams; nunca un modelo por cámara. FP32 está prohibido en producción.
- Los agentes usan la interfaz tipada de `src/bus/hub.py`, nunca el MsgHub crudo.
- Solo Python (sin Rust, Go ni C++ propio). Despliegue con Docker Compose sobre JetPack 6, sin Kubernetes.
- Ante un OOM se baja a 720p o se reduce el batch; no se duplican modelos.

Las decisiones abiertas (ARCH-002 modelo, ARCH-003 NvDCF vs ByteTrack) están en la tabla de decisiones de `docs/architecture.md`. Actualízala cuando se resuelvan.

Los artefactos `*.engine` y `*.onnx` están en `.gitignore`; no se versionan.
