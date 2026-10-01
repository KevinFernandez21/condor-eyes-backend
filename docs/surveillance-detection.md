# Detector general de personas y objetos para CCTV (issue #9)

Detector que alimenta a los agentes de tracking, identidad y eventos con
detecciones de `person` (obligatoria) y objetos relevantes en cámaras fijas.

Código en `src/surveillance/`, CLI en `scripts/surveillance.py`, configuraciones
versionadas en `configs/surveillance*.toml` e informe en
[`docs/reports/surveillance-detection.md`](reports/surveillance-detection.md).

## Clases del primer prototipo

| # | Clase | Por qué |
|---|---|---|
| 0 | `person` | Obligatoria: tracking, identidad, eventos |
| 1–5 | `bicycle`, `car`, `motorcycle`, `bus`, `truck` | Vehículos en accesos y parkings |
| 6–8 | `backpack`, `handbag`, `suitcase` | Objetos que pueden quedar abandonados |

Las nueve existen en COCO, así que `yolov8n.pt` (COCO, 80 clases) se proyecta a
este espacio sin reentrenar. El modelo afinado sale directamente con estas 9.

## Dataset card

| Fuente | Uso | Licencia | Contenido |
|---|---|---|---|
| COCO 2017 train (6000 imágenes con alguna clase + 600 sin ninguna) | Fine-tuning | Anotaciones CC BY 4.0; imágenes de Flickr con sus propias licencias (ver términos de COCO) | Escenas generales |
| COCO 2017 val, `image_id` **par** (1539) | Validación y calibración de umbrales | Igual que COCO | |
| COCO 2017 val, `image_id` **impar** (1470) | **Test** de las 9 clases | Igual que COCO | |
| COCO 2017 val impar **sin ninguna clase** (500) | **Hard negatives**: tasa de falsos positivos | Igual que COCO | Comida, animales, interiores… |
| MOT16 train 04 (primer 70 %), 05, 10, 11, 13; un frame de cada 3; visibilidad ≥ 0,25 | Fine-tuning `person` | CC BY-NC-SA 3.0 (MOTChallenge) | Peatones, cámaras fijas y móviles |
| MOT16-04 (último 30 %) | Validación `person` | Igual | |
| **MOT16-02 y MOT16-09** (cámaras fijas; un frame de cada 2; 563 frames, 11.546 personas) | **Test CCTV** `person`, con visibilidad y altura por caja | Igual | Nunca se usan para entrenar ni calibrar |
| Simuletic CCTV Weapon (141, solo `person`) | Test CCTV sintético | CC BY-SA 4.0 | Ojo de pez, nocturno, exteriores |

**Reglas de split:**

- COCO val se parte por paridad de `image_id`.
- MOT se parte por secuencia. Las dos secuencias de cámara fija quedan reservadas para test.
- Las cajas MOT de clase 1, 2 y 7 (peatón, persona en vehículo, persona estática) con
  `consider=1` cuentan como `person`.
- En test se incluyen también las personas muy ocluidas (visibilidad < 0,25), por eso
  el recall se reporta además por tramos de visibilidad y de altura.

**Distribución de cajas:**

| Split | Detalle |
|---|---|
| train COCO | 30.096 cajas, mayoría `person` |
| train MOT | 19.938 cajas `person` |
| test COCO | 7.444 cajas |
| test MOT | 11.546 cajas `person` |

El detalle por clase está en `datasets/surveillance/prepare_report.json`.

**Robustez:** sobre el test MOT se repite la evaluación con poca luz (gamma 2,5, 60 %
de brillo y ruido) y con desenfoque de movimiento (15 px).

Datasets, pesos, predicciones y videos quedan fuera de Git (`/data`, `/datasets`,
`/runs`, `/reports`, `/weights`).

## Comandos

```bash
uv run python scripts/surveillance.py prepare
uv run python scripts/surveillance.py predict --tag baseline
uv run python scripts/surveillance.py evaluate --tag baseline --write-config configs/surveillance.toml --version v0-coco-baseline
uv run python scripts/surveillance.py train --epochs 40 --workers 2
uv run python scripts/surveillance.py predict --tag finetuned --model runs/surveillance/yolov8n_coco_mot/weights/best.pt
uv run python scripts/surveillance.py evaluate --tag finetuned --model runs/surveillance/yolov8n_coco_mot/weights/best.pt
uv run python scripts/surveillance.py annotate --config configs/surveillance_v1.toml --input data/raw/mot/videos/MOT16-09.mp4 --output reports/surveillance/demo_MOT16-09.mp4
```

- `prepare` baja las 6000 imágenes de COCO train una a una. Las anotaciones y val2017
  van en zips, y MOT16 se baja de motchallenge.net.
- `predict` cachea predicciones con conf 0,001.
- `evaluate` calibra un umbral por clase (F1 máximo en val) y reporta el held-out:
  AP50, AP50-95, precisión/recall al umbral, tasa de falsos positivos en negativos y
  recall por visibilidad y altura.
- `train` usa 2 workers porque con más la RAM no alcanzó en la laptop (OpenBLAS sin
  memoria).

## Contrato con el resto del sistema

`SurveillanceDetector` cumple el `Protocol` `Detector` del repositorio y devuelve
`{"xyxy", "cls", "conf", "label"}` con `cls` en el espacio de las 9 clases. Aplica el
umbral por clase de la config versionada (`version`, `model`, `imgsz`, `[conf]`).

Para la Jetson se exporta un engine TensorRT FP16 del modelo elegido **en la propia
Jetson**; esa validación sigue diferida, como en #1.

## Licencias antes de uso no investigativo

- Ultralytics (YOLOv8): AGPL-3.0, la misma decisión pendiente que en #1.
- COCO: anotaciones CC BY 4.0; las imágenes conservan la licencia de Flickr de cada una.
- MOT16: CC BY-NC-SA 3.0, **no comercial**. Si el modelo afinado se usara fuera de
  investigación habría que reentrenarlo sin MOT. El modelo recomendado (ver informe)
  no usa MOT.
