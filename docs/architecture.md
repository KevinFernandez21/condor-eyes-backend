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
src/bus/hub.py                    # wrapper MsgHub: publish/subscribe tipado
```

## 4. Presupuesto Orin Nano 8GB (1080p)

- NVDEC: ~11x1080p30 H.265 teórico; objetivo real **4-8 streams**.
- Inferencia: YOLOv8n FP16 640px, batch 4-8, un solo engine.
- Memoria unificada 8GB: ~2GB sistema, resto a batch + tracker + buffers.
  Si hay OOM, bajar a 720p o reducir batch, no duplicar modelos.

## 5. Decisiones

| ID | Estado | Descripción |
|---|---|---|
| ARCH-001 | Decidido | Orquestación: AgentScope, todo Python |
| ARCH-002 | Pendiente | Modelo base: YOLOv8n vs PeopleNet en Orin |
| ARCH-003 | Pendiente | Tracker: NvDCF vs ByteTrack liviano |
| ARCH-004 | Decidido | Repo anterior descartado |
| ARCH-005 | Decidido | Video fuera del bus AgentScope, solo metadata por MsgHub |

## 6. Alternativas descartadas

- **Video por mensajes AgentScope**: latencia y memoria inviables en edge.
- **Un proceso YOLO por cámara**: duplica motores y revienta VRAM.
- **Kubernetes**: sobrecarga para un solo Orin. Docker Compose.
