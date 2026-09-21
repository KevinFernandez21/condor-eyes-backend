# Tech stack aprobado — Condor Eye / Orin Nano 8GB / AgentScope + Python

Todo Python. Sin Rust, sin Go, sin C++ propio.

| Capa | Elección | Notas |
|---|---|---|
| Orquestación agentes | AgentScope (`agentscope`, MsgHub, ReActAgent) | Python 3.10+ según JetPack; agentes herramienta para video, ReAct para `event`/`supervisor` |
| Video | GStreamer Python + NVDEC + `nvstreammux` (DeepStream Python si disponible) | El pipeline vive en `src/pipeline/`, fuera del bus de mensajes |
| Inferencia | TensorRT FP16 Python, YOLOv8n o PeopleNet (ARCH-002) | Un solo engine versionado |
| Tracker | NvDCF primero, ByteTrack si falta RAM (ARCH-003) | |
| Metadata agentes | AgentScope MsgHub en proceso | Sin broker externo en Fase 0 |
| Salida exterior | FastAPI en `comms` + SQLite WAL | Postgres solo con servidor externo |
| Env edge | JetPack 6, Docker + Compose | Sin Kubernetes |
| Calidad/review | `ruff`, `mypy`, `pytest`; gates `review validate` en pre-commit/pre-push/pre-pr/release al crear git | Sin reviews dentro de gates, solo validación |

## Reglas

- Ningún frame sale del pipeline al MsgHub: solo detecciones, tracks y eventos.
- FP16 obligatorio en Jetson. Sin FP32 en producción.
- Interfaz bus tipada en `src/bus/hub.py`; ningún agente importa el bus crudo.
