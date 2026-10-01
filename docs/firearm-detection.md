# Prototipo: detección de armas en video grabado (issue #1)

Primer corte vertical de visión: detectar una **pistola/arma de fuego** en un
video grabado y generar un MP4 anotado con caja, clase, confianza y FPS.

> Esto es un **prototipo de detección, no una garantía de seguridad**. Los falsos
> positivos y falsos negativos se reportan tal cual en el informe.

## Alcance y hardware

| | Ahora | A futuro |
|---|---|---|
| Máquina | Laptop x86_64, **RTX 5080 Laptop 16 GB**, Windows 11 | Jetson Orin Nano 8 GB |
| Pesos | `.pt` de ultralytics, FP16 en GPU (`quantize=16`) | Engine TensorRT FP16 exportado y validado en la Jetson |
| Entrenamiento | En la propia laptop | Fuera de la Jetson (sin cambios) |

La validación en la Jetson (export a `.engine`, demo sin OOM, memoria y
latencia en el edge) **queda diferida** a un issue posterior. El código ya
admite un `.engine` en `model` porque ultralytics lo carga igual que un `.pt`.

Fuera de alcance, igual que en el issue: DeepStream, tracking, alertas, RTSP,
multicámara, reconocimiento facial y notificación a autoridades.

## Setup desde un entorno limpio

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run pytest
```

`pyproject.toml` toma `torch`/`torchvision` del índice CUDA 13.0 de PyTorch en
x86_64 (las RTX serie 50 necesitan CUDA ≥ 12.8). En aarch64 (Jetson) ese índice no
aplica; allí se usará el wheel de JetPack.

Todos los comandos pasan por `scripts/firearm.py` (agrega `src/` al `sys.path`):

```bash
uv run python scripts/firearm.py --help
```

## 1. Datos: CCTV-Gun

[CCTV-Gun](https://github.com/srikarym/CCTV-Gun) reúne imágenes de vigilancia de tres
fuentes con anotaciones COCO de dos clases, `person` y `handgun`. Las imágenes
**no** vienen en ese repo: hay que bajarlas a mano de cada fuente original
(Google Drive, OneDrive y Dropbox) siguiendo su
[`dataset_instructions.md`](https://github.com/srikarym/CCTV-Gun/blob/master/dataset_instructions.md),
y ejecutar sus scripts `copy_images_*.py`. El resultado es un `data/` con
`all_images/` y las anotaciones.

Split del prototipo:

| Split | Fuente | Uso |
|---|---|---|
| train / val | `data/mgd_usrt/annotations_{train,val}.json` (MGD + USRT) | Fine-tuning |
| test | `data/ucf/annotation_detection/annotations_all.json` (UCF completo) | Held-out entre dominios, nunca visto en entrenamiento |
| negatives | Carpeta propia (`--negatives`) | Hard negatives: teléfonos, herramientas, manos vacías |

```bash
uv run python scripts/firearm.py prepare-data --root ../CCTV-Gun/data --out datasets/cctv_gun_mgd_usrt --negatives data/hard_negatives
```

Genera `images/`, `labels/` (formato YOLO), `data.yaml` y `prepare_report.json`
con el conteo de imágenes y cajas por split. Las imágenes se enlazan con
hardlinks cuando se puede, para no duplicar espacio.

## 2. Entrenamiento (fine-tuning)

Parte de `yolov8n.pt` preentrenado en COCO, con `imgsz=640`:

```bash
uv run python scripts/firearm.py train --epochs 100 --batch 32
```

Los pesos quedan en `runs/firearm/yolov8n_mgd_usrt/weights/best.pt`, que es el
`model` por defecto de `configs/firearm.toml`. El batch de entrenamiento puede
ser mayor que 1; **la inferencia siempre es batch 1**.

## 3. Evaluación

```bash
uv run python scripts/firearm.py eval --split test --negatives data/hard_negatives --output reports/eval_test.json
```

Reporta mAP50 y mAP50-95 globales y por clase sobre el held-out UCF. También
mide la **tasa de falsos positivos** en hard negatives, es decir, el porcentaje de
imágenes sin arma en las que se detecta `handgun` con `conf ≥ umbral`.

## 4. Video anotado

```bash
uv run python scripts/firearm.py annotate --input demo.mp4 --output reports/demo_annotated.mp4
```

- Cajas rojas para clases de arma (`weapon_classes`) y verdes para el resto, con `clase confianza`.
- Overlay con FPS de extremo a extremo (media móvil) y latencia de inferencia del frame.
- Escribe `reports/demo_annotated.json` con: FPS promedio (extremo a extremo y solo
  inferencia), latencia media/p50/p90/p99, frames con arma, pico de RSS del proceso
  y pico de memoria CUDA (reservada y asignada por torch).
- El warmup se hace con el primer frame real, así la latencia medida no incluye la
  autoconfiguración de kernels CUDA para esa resolución.

Configuración (`configs/firearm.toml`, sección `[detector]`): `model`, `conf`,
`iou`, `imgsz`, `device`, `half`, `classes`, `weapon_classes`. Los flags
`--model --conf --iou --imgsz --device` la sobrescriben.

Desde un notebook, `FirearmDetector` cumple el `Protocol` `Detector`, así que
entra directo en `compare_models`:

```python
import sys; sys.path.insert(0, "src")
from compare import compare_models
from firearm import FirearmDetector, load_config
rows = compare_models([FirearmDetector(load_config("configs/firearm.toml"))], source="demo.mp4", frames=300)
```

## Licencias y procedencia

### Modelo: Ultralytics (AGPL-3.0). **Decisión pendiente**

El código de Ultralytics y los modelos entrenados con él son AGPL-3.0 por
defecto. Antes de que este prototipo pase a ser un entregable propietario o
comercial hay que decidir una de estas opciones:

1. Cumplir AGPL-3.0, es decir, publicar el código del servicio bajo AGPL.
2. Comprar una licencia Ultralytics Enterprise.
3. Si ninguna sirve, evaluar **YOLOX-Nano** (Apache-2.0) en un issue aparte. No se
   implementan ambos modelos en esta rama.

### Datos

| Fuente | Qué es | Licencia / uso | Estado |
|---|---|---|---|
| CCTV-Gun (anotaciones y scripts) | Benchmark de Yellapragada et al., 2023 | Repo bajo Apache-2.0 | Verificado (licencia del repo en GitHub) |
| MGD (Monash Gun Dataset) | Lim et al., 2021 | Términos de la fuente original | **Por verificar antes de descargar** |
| USRT (Universidad de Sevilla) | González et al., 2020 | Términos de la fuente original | **Por verificar antes de descargar** |
| UCF-Crime | Sultani et al., 2018 | Uso de investigación según sus autores | **Por verificar antes de descargar** |
| Hard negatives propios | Fotos/clips sin arma | Según su origen | Registrar el origen de cada imagen |

Que el repo de CCTV-Gun sea Apache-2.0 **no** cubre las imágenes de MGD, USRT
ni UCF: cada fuente tiene sus propios términos. Hay que documentarlos aquí antes
de usar el modelo fuera de un contexto de investigación.

Datasets, pesos, engines, videos y reportes generados **no se versionan**:
`data/`, `datasets/`, `weights/`, `runs/`, `reports/`, `*.pt`, `*.mp4`, `*.engine`
y `*.onnx` están en `.gitignore`. El video demo se adjunta al PR como artefacto
externo.

## Informe

La plantilla para completar con los resultados está en
[`docs/reports/firearm-prototype.md`](reports/firearm-prototype.md).
