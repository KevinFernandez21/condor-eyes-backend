# Arquitectura multiagente — Condor Eye / Orin Nano 8GB / AgentScope + Python

Todo el sistema es Python. La orquestación multiagente usa **AgentScope**
(mensajería, MsgHub, herramientas). El video usa un pipeline Python
compartido (GStreamer + NVDEC + TensorRT) que los agentes controlan pero
no atraviesa el bus de mensajes.

## 1. Agentes (todos `agentscope.Agent`)

| Agente | Tipo AgentScope | Rol |
|---|---|---|
| `ingest` (xN) | agente herramienta | Abre un stream RTSP/USB, reconexión y watchdog, publica estado por MsgHub |
| `inference` (único) | agente herramienta | Consume el batch compartido del pipeline, publica detecciones |
| `tracker` | agente herramienta | NvDCF o ByteTrack por stream, publica tracks |
| `event` | ReActAgent | Reglas sobre tracks (zonas, merodeo, conteo), emite eventos |
| `storage` | agente herramienta | Guarda clips por evento + SQLite |
| `supervisor` | ReActAgent | Salud, reinicios, config, OTA |
| `comms` | agente herramienta | Puente MsgHub → exterior (API HTTP/WS mínima) |

`ingest` escala con N cámaras. `inference` no se replica: escala por
batch-size y resolución.

## 2. Planos separados

```
Plano video (Python, cero copias, fuera de AgentScope):
cámaras ─→ GStreamer NVDEC → nvstreammux (batch N) → nvinfer FP16 → nvtracker → appsink

Plano agentes (AgentScope MsgHub, solo metadata):
ingest ↔ inference ↔ tracker ↔ event → storage / comms
supervisor observa a todos
```

Motivo: MsgHub mueve mensajes, no frames NVMM. Pasar video por AgentScope
rompería el presupuesto de memoria y latencia de la Orin Nano.

## 3. Módulos Python

```
src/agents/ingest_agent.py, inference_agent.py, tracker_agent.py,
          event_agent.py, storage_agent.py, supervisor_agent.py, comms_agent.py
src/pipeline/shared_pipeline.py   # único pipeline batcheado GStreamer
src/pipeline/trt_engine.py        # carga del engine TensorRT FP16 versionado
src/bus/hub.py                    # contrato tipado: Topic, MetadataEnvelope, validación
src/bus/memory.py, agentscope_hub.py  # adaptadores (en memoria / Msg de AgentScope 2.x)
src/agents/runtime.py, handlers.py    # ciclo de vida y lógica por rol
src/actuation/                     # nodo Pan-Tilt ESP32-S3 (ver docs/pan-tilt.md)
```

AgentScope 2.x no incluye `MsgHub`; el bus usa `Msg` y `Agent.observe`. Detalle,
garantías y diagrama de secuencia en [agentscope-runtime.md](agentscope-runtime.md).

## 4. Presupuesto Orin Nano 8GB (1080p)

- NVDEC: ~11x1080p30 H.265 teórico; objetivo real **4-8 streams**.
- Inferencia: YOLOv8n FP16 640px, batch 4-8, un solo engine.
- Memoria unificada 8GB: ~2GB sistema, resto a batch + tracker + buffers.
  Si hay OOM, bajar a 720p o reducir batch, no duplicar modelos.
- Presupuesto medido en la laptop sin CUDA como proxy (OpenVINO FP16, iGPU; issue #23,
  `docs/edge-budget.md`): el stack completo (detector + re-ID + verificación facial) a 640
  con 8 streams 1080p ocupa 3,6 GB (detector unificado) o 4,4 GB (dos detectores); a 1280
  no cabe con margen. Orden ante OOM: entrada 1280→640, cámaras 1080p→720p, sub-lotes,
  menor frecuencia de etapas opcionales, menos streams.

## 5. Decisiones

| ID | Estado | Descripción |
|---|---|---|
| ARCH-001 | Decidido | Orquestación: AgentScope, todo Python |
| ARCH-002 | Decidido | Modelo base: YOLOv8n (ultralytics), descartado PeopleNet |
| ARCH-003 | Pendiente | Tracker: NvDCF vs ByteTrack liviano |
| ARCH-004 | Decidido | Repo anterior descartado |
| ARCH-005 | Decidido | Video fuera del bus AgentScope, solo metadata por MsgHub |
| ARCH-006 | Decidido | Detector: objetivo un único YOLOv8n de 10 clases (9 de vigilancia + `weapon`) en un solo engine FP16 (−40 % de cómputo y −0,2/0,7 GB frente a dos detectores, medido en #23). Requiere reentrenar (issue aparte); mientras tanto, **excepción temporal**: dos engines (vigilancia #9 + armas #1), nunca uno por cámara, con el de armas a menor frecuencia si falta throughput |
| ARCH-007 | Decidido | Precisión: FP16 en todos los modelos (detector, re-ID, YuNet, SFace); INT8 solo con calibración documentada; FP32 prohibido en producción; única excepción registrada: la herramienta de evaluación de #10 en la laptop (OpenCV DNN FP32), no producción (SFace FP16 equivale a FP32: coseno ≥ 0,9998) |
| ARCH-008 | Decidido | Entrada del detector 640 por defecto en multistream; 1280 solo con N ≤ 2 cámaras |
| ARCH-009 | Pendiente | Validar el presupuesto de #23 en la Jetson con TensorRT FP16 + NVDEC (checklist en `docs/edge-budget.md`) |

## 6. Alternativas descartadas

- **Video por mensajes AgentScope**: latencia y memoria inviables en edge.
- **Un proceso YOLO por cámara**: duplica motores y revienta VRAM.
- **Kubernetes**: sobrecarga para un solo Orin. Docker Compose.
