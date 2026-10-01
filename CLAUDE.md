# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Estado del repo

Proyecto en Fase 0: videovigilancia multiagente para **Jetson Orin Nano 8GB**, todo en **Python + AgentScope**, objetivo 4-8 streams 1080p con inferencia compartida. Se realizará un **entorno de campo multiagéntico**: el sistema se despliega y opera en campo, en el edge, como un conjunto de agentes AgentScope coordinados. Reemplaza al repo anterior `condor-eye-backend` (ISR para drones); no se reutiliza nada de él.

Lo implementado es `src/compare/` (harness de benchmark de detectores) y `src/firearm/` (prototipo de detección de armas en video grabado, issue #1; ver `docs/firearm-detection.md`). El modelo base ya está decidido: **YOLOv8n** (ARCH-002).

**Máquina de desarrollo actual: laptop x86_64 con RTX 5080 Laptop 16 GB (Windows).** Entrenamiento, inferencia y demos se validan ahí. La validación en la Jetson Orin Nano (export TensorRT `.engine`, pruebas sin OOM) queda diferida; no bloquea el trabajo actual. Los módulos `src/agents/`, `src/pipeline/` y `src/bus/` descritos en el README y en `docs/architecture.md` **todavía no existen**; son el diseño a seguir al crearlos.

## Entorno y comandos (uv)

El proyecto se gestiona con **uv**: `pyproject.toml` + `uv.lock`, Python **3.11** fijado en `.python-version` (AgentScope 2.x exige >=3.11). Dependencias: `agentscope` 2.x y `ultralytics`; grupo dev: `ruff`, `mypy`, `pytest`.

```bash
uv sync                      # crea/actualiza .venv según uv.lock
uv add <pkg>                 # dependencia nueva (nunca pip install)
uv add --dev <pkg>           # dependencia de desarrollo
uv run ruff check src
uv run mypy src
uv run pytest                # tests en tests/ (pythonpath=src vía pyproject)
uv run pytest tests/test_x.py::test_y   # un solo test
```

No hay Dockerfile ni pre-commit todavía. `FakeDetector` permite probar el harness sin GPU ni pesos.

`torch`/`torchvision` salen del índice CUDA 13.0 de PyTorch en x86_64 (`[tool.uv.sources]`); las RTX serie 50 necesitan CUDA >= 12.8. El CLI del prototipo de armas es `uv run python scripts/firearm.py {split-scenes,prepare-data,train,eval,annotate}`. El dataset en uso es Simuletic CCTV Weapon (Kaggle, sintético, clases `person`/`weapon`, split por escena); CCTV-Gun quedó bloqueado por enlaces caídos. Resultados en `docs/reports/firearm-prototype.md`.

En la Orin, JetPack 6 trae Python 3.10 y sus bindings de TensorRT/DeepStream están compilados para 3.10; con el Python 3.11 de uv habrá que conseguir o compilar esos bindings para 3.11.

La documentación y los mensajes de error están en español; mantén ese idioma.

## Harness de comparación (`src/compare/`)

Está pensado para llamarse desde un notebook del usuario (con el kernel de `.venv`), no como CLI. `pyproject.toml` no define build system, así que `src` no se instala como paquete y hay que añadirlo al `sys.path`:

```python
import sys; sys.path.insert(0, "src")
from compare import YOLOv8nDetector, PeopleNetDetector, FakeDetector, compare_models
rows = compare_models([FakeDetector(), YOLOv8nDetector()], source="video.mp4", frames=100)
```

- Todos los detectores cumplen el `Protocol` `Detector` (`name`, `warmup`, `infer(frame) -> list[dict]`, `close`). Un detector nuevo debe respetarlo para entrar en `compare_models`, que llama a `close()` aunque falle.
- Las dependencias pesadas (`ultralytics`, `tensorrt`, `onnxruntime`, `cv2`) se importan de forma diferida dentro de los métodos, para que el módulo cargue en cualquier máquina. Mantén ese patrón.
- `source` en `benchmark_detector` admite un callable, una lista de frames, una ruta de video (cv2) o `None`.
- `YOLOv8nDetector` es el detector del proyecto; `weights` admite `.pt` o el engine TensorRT FP16 exportado con ultralytics.
- `PeopleNetDetector` queda como referencia descartada: la ruta `.onnx` devuelve solo las formas de salida y la ruta TensorRT lanza `NotImplementedError`. No invertir trabajo en él.

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

La decisión abierta (ARCH-003 NvDCF vs ByteTrack) están en la tabla de decisiones de `docs/architecture.md`. Actualízala cuando se resuelvan.

Los artefactos `*.engine` y `*.onnx` están en `.gitignore`; no se versionan.
