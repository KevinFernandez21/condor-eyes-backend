# Informe: detector general de personas y objetos (issue #9)

Corrida del 2026-10-01 en la laptop (RTX 5080 Laptop 16 GB). Splits y licencias en
`docs/surveillance-detection.md`. Los umbrales por clase se calibraron en val (F1
máximo) y **nunca** con test.

## Candidatos

| Versión | Modelo | Entrada | Entrenamiento |
|---|---|---|---|
| `v0-coco-baseline` | `yolov8n.pt` (COCO) | 640 | Ninguno; umbrales por clase calibrados |
| **`v1-coco-1280`** | `yolov8n.pt` (COCO) | **1280** | Ninguno; umbrales recalibrados a 1280 |
| `finetuned` | YOLOv8n afinado con COCO (6000) + MOT16 (1292 frames), 40 épocas, lr 0,002 | 640 | 47 min en la laptop |

## Resultados en held-out

| Test | Métrica | v0 (640) | **v1 (1280)** | Afinado (640) |
|---|---|---|---|---|
| COCO val impar, 9 clases | mAP50 / mAP50-95 | **0,497** / **0,328** | 0,487 / 0,295 | 0,397 / 0,246 |
| **MOT16-02 + 09 (CCTV fija)**, `person` | AP50 / AP50-95 | 0,465 / 0,264 | **0,548** / **0,316** | 0,491 / 0,279 |
| | Precisión / recall al umbral | 0,69 / 0,37 | 0,76 / **0,46** | **0,82** / 0,36 |
| Simuletic (sintético CCTV), `person` | AP50 / AP50-95 | 0,968 / 0,713 | **0,981** / **0,738** | 0,964 / 0,732 |
| 500 negativos COCO | Imágenes con falso positivo | **8,0 %** | 8,8 % | 14,8 % |

AP50 por clase en COCO (v0 / v1):

| Clase | v0 | v1 |
|---|---|---|
| person | 0,75 | 0,75 |
| bicycle | 0,44 | 0,46 |
| car | 0,60 | 0,66 |
| motorcycle | 0,68 | 0,57 |
| bus | 0,70 | 0,68 |
| truck | 0,41 | 0,41 |
| backpack | 0,20 | 0,21 |
| handbag | 0,18 | 0,18 |
| suitcase | 0,52 | 0,47 |

Los **objetos pequeños de mano** (mochila, bolso) son débiles en todos los modelos
(AP50 de 0,18–0,21). No sirven todavía para "objeto abandonado" sin más datos.

### El fine-tuning no compensa

En val-MOT el afinado llegó a AP50 0,915, pero en el held-out MOT solo sube de 0,465 a
0,491. Val-MOT sale de la misma secuencia que el entrenamiento (MOT16-04), así que mide
memorización de la escena, no generalización. Además, el afinado pierde 0,10 de mAP50 en
COCO por olvidar clases y duplica los falsos positivos en negativos (detecta maletas
falsas 27 veces). También arrastra la licencia no comercial de MOT16.

### Lo que sí ayuda: más resolución de entrada

En CCTV 1080p, las personas lejanas quedan en un tercio de su tamaño a 640 px. A 1280,
el AP50 de `person` en MOT pasa de 0,465 a 0,548 sin entrenar nada. El coste es algo de
mAP en COCO, donde las imágenes son de unos 640 px y no ganan nada.

## Análisis de fallos (MOT held-out, recall de `person`)

| Visibilidad (oclusión) | Cajas | v0 | **v1** |
|---|---|---|---|
| < 0,25 (casi oculta) | 5040 | 6,8 % | 13,4 % |
| 0,25–0,5 | 1322 | 26,1 % | 40,5 % |
| 0,5–0,75 | 1223 | 45,5 % | 59,7 % |
| ≥ 0,75 (visible) | 3961 | 74,5 % | **82,9 %** |

| Altura de la persona | Cajas | v0 | **v1** |
|---|---|---|---|
| < 50 px (muy lejana) | 607 | 0 % | 1,2 % |
| 50–100 px | 3404 | 8,0 % | 19,8 % |
| 100–200 px | 3405 | 24,4 % | 40,7 % |
| ≥ 200 px | 4130 | 74,9 % | 76,4 % |

**Robustez** (v0, AP50 de `person` en MOT):

| Condición | AP50 |
|---|---|
| Original | 0,465 |
| Poca luz (gamma 2,5, 60 % de brillo y ruido) | 0,380 |
| Desenfoque de movimiento (15 px) | 0,381 |

En ambas el recall cae y la precisión se mantiene.

Ejemplos revisados a mano (imágenes en `reports/surveillance/failures_*.jpg`, fuera de Git):

- **Oclusión en grupos** (MOT16-02, frame 401): 20 de las 21 personas perdidas tienen
  visibilidad < 0,25 y están detrás de otras. El 44 % de las cajas del test MOT son así.
  Es el límite principal del recall global, más que un fallo corregible.
- **Reflejos** (MOT16-09, frame 301): los "falsos positivos" a la izquierda son personas
  reflejadas en una vitrina, que MOT no etiqueta. Un tracker con zonas de exclusión las
  filtraría.
- **Personas pequeñas y lejanas:** por debajo de 100 px de alto casi no se detectan a
  640 px. Es la razón de usar 1280.
- **Poca luz y desenfoque:** en ambos casos el AP50 baja unos 8 puntos.

## Rendimiento (batch 1, `annotate` sobre MOT16-09, 1920×1080, 525 frames)

| Versión | FPS extremo a extremo | FPS inferencia | Latencia p50 / p90 | Pico CUDA reservado | Pico RSS |
|---|---|---|---|---|---|
| v0 (640) | 38,9 | 97,2 | 9,7 / 12,3 ms | 36 MB | 1,9 GB |
| **v1 (1280)** | 33,5 | 70,6 | 13,8 / 16,5 ms | 68 MB | 1,9 GB |

- El extremo a extremo incluye decodificar, dibujar y codificar el video 1080p.
- La resolución de entrada 1280 cuadruplica los píxeles del modelo. En la Jetson
  (4–8 streams) hay que medirlo; si no cabe, la regla del repo es bajar la
  resolución o el batch, no duplicar modelos.
- Demo: `reports/surveillance/demo_MOT16-09_surveillance_v1.mp4` (17,5 s, fuera de Git),
  con caja, clase, confianza y FPS.

## Decisión

1. **Configuración versionada recomendada: `configs/surveillance_v1.toml`.** YOLOv8n
   COCO **sin reentrenar** a 1280, con umbrales por clase calibrados en val. Es el mejor
   en CCTV, no arrastra la licencia de MOT y mantiene los falsos positivos en 8,8 %.
2. `configs/surveillance.toml` (v0, 640) queda como alternativa cuando el presupuesto de
   la Jetson no permita 1280.
3. **No usar el modelo afinado.** Para mejorar de verdad hacen falta frames **del sitio**
   con escenas y cámaras distintas en val y test, más ejemplos de mochilas y bolsos.
4. **Pendiente en la Jetson:** exportar a TensorRT FP16 en el dispositivo
   (`yolo export format=engine half=True imgsz=1280`) y medir FPS y memoria con varios
   streams.
