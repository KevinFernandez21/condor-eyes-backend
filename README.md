# Condor Eye — videovigilancia multiagente en Jetson Orin Nano 8GB

Reemplazo total del repo anterior (`condor-eye-backend`). Nada del scaffold
ISR para drones se reutiliza.

Target único: **Jetson Orin Nano 8GB**. Todo en **Python + AgentScope**.
Diseño para **multistream 4-8x 1080p** con inferencia compartida.

## Estructura

```
.
├── docs/
│   ├── architecture.md   # agentes AgentScope, pipeline, flujos
│   └── tech-stack.md     # stack Python aprobado
├── src/
│   ├── agents/           # un módulo por agente AgentScope
│   ├── pipeline/         # pipeline video Python compartido
│   └── bus/              # wrapper MsgHub AgentScope
└── README.md
```

## Principios

- Todos los agentes son agentes AgentScope en Python.
- Video pesado en un solo pipeline Python compartido (GStreamer + NVDEC);
  AgentScope orquesta mensajes y eventos, no mueve frames.
- Inferencia batcheada y compartida (un engine TensorRT FP16, N streams).
- Cada agente es reemplazable sin tocar el resto.
- Edge con Docker Compose. Sin Kubernetes.
