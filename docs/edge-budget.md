# Presupuesto de memoria y throughput: laptop como proxy de la Jetson (issue #23)

Objetivo: un único presupuesto de memoria y throughput para **todos** los modelos
que proponen #1, #9, #10 y #11, en lugar de que cada uno recomiende su configuración
por separado. El destino sigue siendo la **Jetson Orin Nano 8 GB**. Como todavía no
está disponible, se mide en una laptop **sin CUDA**, como proxy de la memoria
unificada, y la validación con TensorRT queda pendiente.

## Hardware y runtime medidos

| Campo | Valor |
|---|---|
| Máquina | Laptop Intel **Core Ultra 9 275HX** (24 hilos), **32 GB** de RAM, Windows 11 (build 26200) |
| Aceleradores usados | **CPU** y **GPU integrada Intel Graphics** (iGPU, memoria compartida con la RAM, como la Jetson) |
| No usados a propósito | GPU dedicada RTX 5080, CUDA, TensorRT, NVDEC, NPU |
| Runtime | **OpenVINO 2026.4.1**, modelos IR FP16; detector "lean" en OpenVINO puro (`src/edge/ovdetect.py`), sin torch en el proceso medido |
| Video | Decodificación por software (OpenCV 5.0.0 / FFmpeg), H.264 1080p30 (MOT16-02 y MOT16-09) y su versión 720p |
| Python | 3.11.16 |
| Fecha | 2026-10-01 |

**No es el hardware del equipo.** La issue describe una laptop i7 de 12.ª generación con
16 GB y GPU integrada. Esta máquina es más rápida. La **memoria** sirve como
referencia, porque depende de modelos, buffers y runtime y no de la velocidad. Los
**FPS no**: hay que repetir la matriz en el i7 con el mismo script.

**Contención durante la medición:** dos entrenamientos de MATLAB ajenos al proyecto
ocupaban la CPU. La carga total del sistema fue del 80–98 % (columna
`system_cpu_percent` de `reports/edge/budget.json`). Los FPS son **pesimistas**,
sobre todo en el dispositivo CPU. Las memorias no se ven afectadas.

**Cómo se mide** (`scripts/edge_budget.py`, `src/edge/budget.py`):

- Cada configuración corre en un **proceso nuevo**.
- N streams son N decodificadores en el mismo proceso.
- **Un solo modelo compilado por detector** procesa los N frames de cada ciclo en un
  lote; nunca un modelo por cámara.
- Memoria:
  - pico de RAM privada (*commit*) y de *working set* del proceso;
  - pico de memoria de GPU del proceso (contadores de Windows `GPU Process Memory`,
    compartida + dedicada).
  - "Total" = RAM privada + GPU. En la iGPU puede contar dos veces parte de la
    memoria, así que es una **cota superior**.
- FPS por stream de extremo a extremo (decodificación + inferencia + recortes) y
  latencia por ciclo p50/p90. Un ciclo es un frame de cada stream.

```bash
uv sync
uv run python scripts/edge_budget.py export --reid <resnet18_market.pt> --faceid <carpeta con YuNet/SFace ONNX>
uv run python scripts/edge_budget.py matrix --output reports/edge/budget.json
```

## Decisión: detector unificado o separado (ARCH-006)

Hoy en `main` hay **dos** detectores YOLOv8n:

- vigilancia (#9): COCO, 9 clases;
- armas (#1): `person` + `weapon`.

Eso choca con la regla de "un solo engine compartido". "Unificado" se mide con un
solo YOLOv8n, que tiene el mismo costo que un modelo de 10 clases porque la
arquitectura no cambia.

| iGPU, 640, 1080p | N = 1 | N = 4 | N = 8 |
|---|---|---|---|
| Unificado: FPS por stream / total GB | 31,4 / 1,32 | 9,8 / 1,99 | 5,2 / 2,84 |
| Separados (vigilancia + armas): FPS por stream / total GB | 17,7 / 1,52 | 5,8 / 2,44 | 3,1 / 3,58 |
| **Costo de separar** | −44 % FPS, +0,20 GB | −41 % FPS, +0,45 GB | −40 % FPS, +0,74 GB |

**Decisión:**

- **Objetivo: un único YOLOv8n de 10 clases** (las 9 de vigilancia + `weapon`),
  entrenado con los datos de #9 y #1, en un solo engine FP16. Ahorra ~40 % de
  cómputo de detección y 0,2–0,7 GB.
- Reentrenar está fuera del alcance de esta issue, así que queda como **issue
  aparte**.
- **Mientras tanto, excepción temporal registrada:** dos engines, nunca uno por
  cámara. Si la Jetson no llega a la meta de FPS, el detector de armas corre a menor
  frecuencia (cada k frames) y no a la de todos los streams.

## Tabla de presupuesto (medido)

Columnas:

- **Disp.:** dispositivo. iGPU = Intel Graphics, `GPU.0`; CPU = Core Ultra 9 275HX.
- **FPS/st:** FPS por stream.
- **Total:** RAM privada + GPU, en GB.

Todos los modelos en OpenVINO FP16 IR, con batch = N streams.

### Detectores (1080p)

| Configuración | Disp. | Entrada | N | FPS/st | p50 / p90 ms | RAM privada | GPU | **Total** |
|---|---|---|---|---|---|---|---|---|
| Unificado | iGPU | 640 | 1 | 31,4 | 32 / 34 | 1,23 | 0,08 | 1,32 |
| Unificado | iGPU | 640 | 4 | 9,8 | 101 / 113 | 1,80 | 0,19 | 1,99 |
| Unificado | iGPU | 640 | 8 | 5,2 | 192 / 206 | 2,52 | 0,32 | 2,84 |
| Unificado | iGPU | 1280 | 1 | 12,5 | 80 / 85 | 1,42 | 0,19 | 1,61 |
| Unificado | iGPU | 1280 | 4 | 3,3 | 301 / 320 | 2,49 | 0,58 | 3,08 |
| Unificado | iGPU | 1280 | 8 | 1,3 | 650 / 976 | 4,01 | 1,19 | 5,20 |
| Separados | iGPU | 640 + 640 | 1 | 17,7 | 56 / 62 | 1,39 | 0,14 | 1,52 |
| Separados | iGPU | 640 + 640 | 4 | 5,8 | 173 / 185 | 2,11 | 0,34 | 2,44 |
| Separados | iGPU | 640 + 640 | 8 | 3,1 | 324 / 359 | 3,00 | 0,58 | 3,58 |
| Separados | iGPU | 1280 + 640 | 4 | 2,8 | 362 / 378 | 2,79 | 0,73 | 3,52 |
| Separados | iGPU | 1280 + 640 | 8 | 1,3 | 717 / 1185 | 4,35 | 1,39 | **5,74** |
| Unificado | CPU | 640 | 4 | 8,7 | 113 / 126 | 1,53 | — | 1,53 |
| Unificado | CPU | 640 | 8 | 4,7 | 212 / 226 | 2,19 | — | 2,19 |
| Separados | CPU | 640 + 640 | 8 | 2,7 | 364 / 399 | 2,47 | — | 2,47 |
| Unificado | CPU | 1280 | 8 | 1,4 | 716 / 756 | 3,29 | — | 3,29 |

### Alternativa a 720p (N = 8)

| Configuración | Disp. | FPS/st 720p (1080p) | **Total 720p (1080p)** |
|---|---|---|---|
| Unificado 640 | iGPU | 6,5 (5,2) | 2,32 (2,84) |
| Separados 640 + 640 | iGPU | 3,5 (3,1) | 3,03 (3,58) |
| Unificado 640 | CPU | 5,1 (4,7) | 1,70 (2,19) |

### Stack completo: detector(es) + re-ID ResNet18 + verificación facial YuNet/SFace (1080p, 640)

Hasta 4 personas por frame pasan a re-ID y una cabeza por stream a la verificación
facial.

| Configuración | Disp. | N | FPS/st | p50 ms | Desglose medio ms (decodificación / detección / re-ID / facial) | **Total GB** |
|---|---|---|---|---|---|---|
| Stack unificado | iGPU | 4 | 7,4 | 134 | 16 / 72 / 28 / 18 | **2,54** |
| Stack unificado | iGPU | 8 | 3,7 | 270 | 32 / 146 / 54 / 35 | **3,59** |
| Stack separados | iGPU | 4 | 5,2 | 193 | 16 / 132 / 27 / 18 | **2,97** |
| Stack separados | iGPU | 8 | 2,5 | 405 | 34 / 278 / 58 / 35 | **4,35** |
| Stack unificado | CPU | 8 | 3,1 | 325 | 31 / 179 / 89 / 26 | 2,54 |
| Stack separados | CPU | 8 | 2,0 | 486 | 32 / 343 / 91 / 27 | 2,82 |

**Criterio de la issue:** el stack completo cabe en **8 GB** con al menos 4 streams
1080p en la laptop. Lo cumple con holgura incluso en el peor caso, separados con N = 8,
que ocupa 4,35 GB.

Lo que **no** cabe con margen es la entrada a **1280 con 8 streams**: 5,2–5,7 GB solo
con los detectores. Con re-ID, facial y el sistema operativo superaría el
presupuesto (ver la proyección).

Cuánto cuesta cada pieza, deducido de las filas anteriores en la iGPU:

- cada stream 1080p decodificado por software: ~0,2 GB;
- cada detector extra: 0,2–0,7 GB según N;
- re-ID + facial: ~0,5 GB a N = 4 y ~0,8 GB a N = 8.

## Precisión por modelo

| Modelo | Laptop (medido) | Jetson (objetivo) | Estado |
|---|---|---|---|
| Detector(es) YOLOv8n | OpenVINO FP16 IR; en la iGPU, cómputo FP16. En CPU, OpenVINO ejecuta los pesos FP16 en FP32/BF16 (solo proxy, no producción) | TensorRT FP16 | FP16; INT8 solo tras calibrar con video del sitio |
| Re-ID ResNet18 (#11) | OpenVINO FP16: coseno con FP32 ≥ 0,99999 | TensorRT FP16 | FP16 |
| YuNet (#10) | OpenVINO FP16 | TensorRT FP16 u OpenCV DNN CUDA FP16 | FP16 en producción; la misma excepción de evaluación que SFace |
| SFace (#10) | OpenVINO FP16: coseno con FP32 ≥ 0,9998 (60 caras DigiFace) | TensorRT FP16 u OpenCV DNN CUDA FP16 | FP16 en producción. **Excepción registrada:** la herramienta de evaluación de #10 (`src/faceid/engine.py`, `faceid.py benchmark`) usa OpenCV DNN FP32 en CPU, solo en la laptop y nunca en producción. Está justificada porque la equivalencia FP16 está medida |
| Eventos (#12) | Python puro, sin modelo | Igual | No aplica |

**Política:** FP16 en todos los modelos. INT8 solo con calibración documentada y un
held-out que demuestre que no empeora. **FP32 prohibido en producción**. La única
excepción vigente es la de la herramienta de evaluación de #10 en la laptop, justificada
arriba.

## Proyección a la Jetson Orin Nano 8 GB (**estimación**, no medida)

| Concepto | Valor | Origen |
|---|---|---|
| Memoria total | 8 GB unificada | Especificación |
| Sistema (JetPack, sin escritorio) | ~1,5–2 GB | Estimación (`docs/architecture.md` §4 usa ~2 GB) |
| **Disponible para la aplicación** | **~6 GB** | 8 − 2 |
| Stack unificado, 640, N = 8, en la laptop | 3,59 GB | **Medido** |
| Stack separados, 640, N = 8, en la laptop | 4,35 GB | **Medido** |
| Detectores separados, 1280, N = 8, en la laptop | 5,74 GB, más ~0,8 de re-ID y facial ≈ 6,5 GB | **Medido + estimado** |

**Lectura:**

- Con entrada **640**, el stack completo de 8 streams queda por debajo de los ~6 GB
  disponibles, con margen (unificado 3,6 GB, separados 4,4 GB). En la Jetson, NVDEC
  con buffers NVMM y TensorRT deberían bajar la memoria de decodificación y de
  runtime, pero **eso es una suposición a validar**.
- Con entrada **1280** y 8 streams, el stack **no cabe** con margen (~6,5 GB estimados).
- **FPS:** no se proyectan. La laptop no tiene TensorRT ni NVDEC, y la CPU estaba
  ocupada. La meta de throughput se mide en la Jetson. Como referencia:
  - 8 streams a 5 FPS con un detector requieren 40 inferencias por segundo a 640;
  - con dos detectores, 80.

## Configuración recomendada y orden ante falta de memoria (OOM)

**Por defecto:**

- un solo engine de detección FP16 a **640** (unificado cuando exista; mientras
  tanto, separados con el de armas a menor frecuencia si hace falta);
- lote = número de streams;
- re-ID solo en los cambios de cámara (no en cada frame) y verificación facial solo
  en los puntos de control;
- 4–8 streams 1080p.

**1280** queda para pocas cámaras (N ≤ 2) donde las personas lejanas importen. Esto
matiza la recomendación de #22 (v1-1280): sirve para precisión en N bajo, no como
valor por defecto para multistream.

**Orden de recorte ante OOM**, sin duplicar nunca modelos:

1. Bajar la entrada del detector de 1280 a **640**.
2. Bajar la resolución de las cámaras de 1080p a **720p**: −0,5 GB a N = 8.
3. **Reducir el lote**: procesar los streams en sub-lotes o bajar los FPS por stream.
4. Bajar la frecuencia de las etapas opcionales: armas cada k frames, re-ID solo en
   cambios de cámara, facial solo en el punto de control.
5. Reducir el número de streams por dispositivo.

## Pendiente en la Jetson (TensorRT FP16), sin dar nada por hecho

- [ ] Exportar en la propia Jetson cada modelo a TensorRT FP16: detector(es) 640/1280,
      re-ID ResNet18, YuNet y SFace.
- [ ] Repetir la matriz con NVDEC + `nvstreammux` (lote N) + `nvinfer`: N = 1, 4, 8 a
      640 y 1280, 1080p y 720p.
- [ ] Medir la memoria con `tegrastats` o jtop (RAM + memoria de GPU unificada),
      frente al límite de 8 GB.
- [ ] Medir FPS de extremo a extremo y latencia p50/p90 por stream, y la meta de FPS
      por cámara.
- [ ] Verificar que FP16 no cambia las métricas de precisión de #1, #9, #10 y #11.
- [ ] Confirmar o corregir el orden de recorte ante OOM con datos reales.
- [ ] Repetir esta matriz en la laptop i7 de 12.ª gen. del equipo para tener la
      referencia de su hardware.
