# Informe: prototipo de detección de armas (issue #1)

Corrida del 2026-09-30 en la laptop de desarrollo. **Prototipo de detección, no
una garantía de seguridad.** Los falsos negativos son altos en video real (ver abajo).

## Entorno

| Campo | Valor |
|---|---|
| Máquina | Laptop, NVIDIA GeForce RTX 5080 Laptop GPU 16 GB, Windows 11 |
| Python / torch / ultralytics | 3.11.16 / 2.14.1+cu130 / 8.4.170 |
| Código | rama `feat/firearm-video-detection` |

## Modelo y datos

| Campo | Valor |
|---|---|
| Modelo | YOLOv8n (`yolov8n.pt` preentrenado en COCO), fine-tuning completo |
| Dataset | Simuletic CCTV Weapon, Kaggle v4 (141 imágenes sintéticas, 6 escenas, CC BY-SA 4.0) |
| Split | por escena: train Scene1–4 (108 imgs), val Scene5 (18), test Scene6 (15) |
| Clases | `person`, `weapon` (armas de fuego y blancas) |
| Entrenamiento | 150 épocas, batch 16, `imgsz=640`, optimizer `auto`, `lr0=0.01`, `seed=0`; 3,3 min. mejor mAP50 en val en la época 129; `best.pt` lo elige ultralytics por fitness de val |
| Inferencia | batch 1, FP16 (`quantize=16`), `conf=0.35`, `iou=0.5`, `imgsz=640` |
| Pesos | `runs/firearm/yolov8n_simuletic/weights/best.pt` (6,2 MB, fuera de Git) |

CCTV-Gun no se pudo usar: el `MGD.rar` da 404, USRT pide login y UCF solo se
baja como la carpeta completa de Dropbox (ver `docs/firearm-detection.md`).

## Precisión

| Split | Clase | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| val (Scene5) | weapon | 0,925 | 0,692 | 0,881 | 0,481 |
| val (Scene5) | person | 0,996 | 0,889 | 0,980 | 0,691 |
| **test (Scene6)** | **weapon** | **0,497** | **0,467** | **0,491** | **0,119** |
| test (Scene6) | person | 1,000 | 0,791 | 0,856 | 0,662 |

La escena de test es una calle vista desde lejos, con armas de pocos píxeles. Por
eso `weapon` cae tanto respecto a val. Con una sola escena en val y otra en test,
estas cifras tienen mucha varianza.

### Comparación `imgsz=960`

Se entrenó la misma receta a 960 (batch 8; parada temprana en la época 101). En
test, `weapon` quedó con mAP50 0,314 (P 0,326, R 0,267), peor que a 640. En el
video detectó el arma en 6 frames, frente a 18. **Se mantiene 640.**

## Video de evaluación (`evaluation.mp4` de Simuletic, 6 s, 464×688, 24 fps, 145 frames)

Es metraje de aspecto real (una persona camina con un rifle en un pasillo), no sintético.

| Tramo | Contenido | Resultado con `conf=0.35` |
|---|---|---|
| frames 0–38 | persona sin arma visible (escena negativa) | 0 falsos positivos de `weapon` |
| frames 39–144 | rifle visible | `weapon` detectado en 18 de 106 frames (≈17 %), todos entre 117 y 144, conf 0,37–0,61 |

`person` se detecta en 138 de 145 frames. Con `conf=0.2`, `weapon` aparece en 37
frames (el primero es el 47) y sigue sin falsos positivos en el tramo negativo. El
problema dominante son los **falsos negativos**: el modelo, entrenado solo con datos
sintéticos, no reconoce el rifle mientras la persona está lejos o de espaldas.

## Rendimiento (batch 1, `annotate` sobre `evaluation.mp4`, GPU sin otra carga)

Tres corridas:

| Métrica | Corrida 1 | Corrida 2 | Corrida 3 |
|---|---|---|---|
| FPS extremo a extremo (decodificar + inferir + dibujar + codificar) | 81,6 | 71,3 | 80,1 |
| FPS solo inferencia | 97,3 | 84,8 | 95,6 |
| Latencia p50 / p90 | 9,2 / 11,8 ms | 10,9 / 13,0 ms | 8,8 / 12,6 ms |
| Pico RSS del proceso | 1843 MB | 1843 MB | 1843 MB |
| Pico memoria CUDA reservada por torch | 36 MB | 36 MB | 36 MB |

La latencia está dominada por el overhead de Python y ultralytics (pre y
postproceso). El modelo en sí tarda menos de 1 ms por imagen en esta GPU.

## Demo

- `reports/evaluation_annotated_640.mp4` (fuera de Git): incluye un tramo negativo y detecciones correctas, pero dura **6 s**.
- **Pendiente:** un demo de ≥ 30 s como pide el issue. Hace falta metraje real más largo con escenas positivas y negativas.

## Conclusiones y siguientes pasos

1. El flujo completo funciona en la laptop: datos → entrenamiento → evaluación → video anotado con métricas.
2. El modelo **no sirve todavía como detector fiable**: tiene recall bajo en video real y en objetos pequeños.
3. Para mejorarlo hacen falta datos reales: conseguir CCTV-Gun (pedírselo a los autores) u otro dataset real con licencia clara, más hard negatives (teléfonos, herramientas, paraguas).
4. Pendiente en la Jetson Orin Nano 8 GB: exportar `best.pt` a TensorRT FP16 en la propia Jetson y repetir `annotate` midiendo FPS, p50/p90 y memoria sin OOM.
