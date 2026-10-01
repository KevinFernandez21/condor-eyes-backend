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

## 1. Datos

### Dataset actual: Simuletic CCTV Weapon (Kaggle)

[`simuletic/cctv-weapon-dataset`](https://www.kaggle.com/datasets/simuletic/cctv-weapon-dataset)
(versión 4, 2025-11-23) es un set **sintético** de 141 imágenes de cámaras CCTV en 6 escenas
(pasillo, patio escolar, gasolinera, parking, restaurante, calle) con etiquetas YOLO
`0 = person` y `1 = weapon`. `weapon` agrupa armas de fuego (sobre todo rifles y
pistolas) y armas blancas, así que el modelo detecta "arma", no solo "pistola".
Trae además `evaluation.mp4` (6 s, 464×688), que se usa como video de prueba.

```bash
mkdir -p data/raw && curl -L -o data/raw/cctv-weapon-dataset.zip https://www.kaggle.com/api/v1/datasets/download/simuletic/cctv-weapon-dataset
unzip data/raw/cctv-weapon-dataset.zip -d data/raw/cctv-weapon
uv run python scripts/firearm.py split-scenes
```

Los frames de una misma escena son casi idénticos, así que `split-scenes` divide
**por escena** para que val y test no compartan escenas con train:

| Split | Escenas | Imágenes | Cajas `weapon` |
|---|---|---|---|
| train | Scene1–Scene4 | 108 | 102 |
| val | Scene5 | 18 | 13 |
| test | Scene6 | 15 | 15 |

Limitaciones: es poco volumen, todo sintético y con pocas escenas. Val y test
tienen una sola escena cada uno, así que las métricas tienen mucha varianza y no
predicen el rendimiento en video real. Sirve para validar el flujo de punta a
punta, no como modelo final.

### CCTV-Gun (bloqueado)

[CCTV-Gun](https://github.com/srikarym/CCTV-Gun) era el dataset recomendado por el
issue (imágenes reales; train MGD+USRT, test UCF). El 2026-09-30 no se pudo
obtener: el `MGD.rar` de Google Drive devuelve 404, el zip de USRT en el
SharePoint de la U. de Sevilla exige login (401) y UCF solo se descarga como la
carpeta completa de Dropbox, de decenas de GB. Si se consigue (por ejemplo,
pidiéndolo a los autores), `prepare-data` ya lo convierte:

```bash
uv run python scripts/firearm.py prepare-data --root ../CCTV-Gun/data --out datasets/cctv_gun_mgd_usrt --negatives data/hard_negatives
```

En ese caso se pasa `--data datasets/cctv_gun_mgd_usrt/data.yaml` a `train`/`eval`
y se cambia `classes`/`weapon_classes` a `handgun` en la config.

## 2. Entrenamiento (fine-tuning)

Parte de `yolov8n.pt` preentrenado en COCO, con `imgsz=640`:

```bash
uv run python scripts/firearm.py train --base weights/yolov8n.pt
```

Los pesos quedan en `runs/firearm/yolov8n_simuletic/weights/best.pt`, que es el
`model` por defecto de `configs/firearm.toml`. El batch de entrenamiento puede
ser mayor que 1; **la inferencia siempre es batch 1**.

## 3. Evaluación

```bash
uv run python scripts/firearm.py eval --split test --negatives data/hard_negatives --output reports/eval_test.json
```

Reporta mAP50 y mAP50-95 globales y por clase sobre el split `test`. También
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
| Simuletic CCTV Weapon (Kaggle, v4) | 141 imágenes sintéticas, etiquetas YOLO | CC BY-SA 4.0 según Kaggle; la descripción del dataset dice CC BY 4.0. Se toma la más restrictiva (BY-SA): atribuir a Simuletic y compartir derivados del dataset con la misma licencia | Verificado (metadatos de Kaggle) |
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
