# Informe: prototipo de detección de armas (issue #1)

> Plantilla. Rellenar con los valores de `reports/*.json` tras entrenar, evaluar y
> anotar el demo. Los campos `TODO` siguen pendientes.

## Entorno

| Campo | Valor |
|---|---|
| Máquina | Laptop, NVIDIA GeForce RTX 5080 Laptop GPU 16 GB, Windows 11 |
| Python / torch / ultralytics | 3.11 / 2.14.1+cu130 / TODO (`uv run python -c "import ultralytics; print(ultralytics.__version__)"`) |
| Commit | TODO |

## Modelo y datos

| Campo | Valor |
|---|---|
| Modelo base | `yolov8n.pt` (COCO), `imgsz=640` |
| Dataset | CCTV-Gun, commit TODO; train/val = MGD + USRT, test = UCF completo |
| Hard negatives | TODO: n imágenes y origen |
| Fine-tuning | TODO: épocas, batch de entrenamiento, early stopping, duración |
| Clases | `person`, `handgun` |

## Precisión (held-out UCF, `eval`)

| Clase | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| handgun | TODO | TODO | TODO | TODO |
| person | TODO | TODO | TODO | TODO |

Falsos positivos en hard negatives: TODO / TODO imágenes (TODO %), con `conf = TODO`.

## Rendimiento (batch 1, `annotate` sobre el demo)

| Métrica | Valor |
|---|---|
| Resolución del video | TODO |
| FPS promedio extremo a extremo | TODO |
| FPS solo inferencia | TODO |
| Latencia p50 / p90 | TODO ms / TODO ms |
| Pico RSS del proceso | TODO MB |
| Pico memoria CUDA (reservada) | TODO MB |

## Demo

- Enlace al video (≥ 30 s, fuera de Git): TODO
- Escenas con arma detectada correctamente: TODO (timestamps)
- Escenas negativas sin arma: TODO (timestamps)
- Falsos positivos y falsos negativos observados: TODO (timestamps y descripción)

## Pendiente para la Jetson Orin Nano 8 GB

- Exportar `best.pt` a TensorRT FP16 en la propia Jetson (`yolo export format=engine half=True imgsz=640`).
- Repetir `annotate` con el `.engine` y registrar FPS, p50/p90 y memoria sin OOM.
