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

## v2: Simuletic + Open Images (2026-10-01)

- **Modelo:** YOLOv8n COCO afinado con Simuletic (108 imágenes) + Open Images V7 (2194
  con arma y 1499 negativos), 640 px, 60 épocas, batch 16, 37 min. La configuración por
  defecto (`configs/firearm.toml`) ya apunta a `runs/firearm/yolov8n_firearm_v2/weights/best.pt`.
- **Comparación** en los mismos tests (`weights/compare_firearm.sh`, umbral `conf=0.35`):

| Test (`weapon`) | v1 (solo Simuletic) | **v2 (+ Open Images)** |
|---|---|---|
| Simuletic Scene6 (CCTV sintético, held-out): precisión / recall | 0,497 / 0,467 | **1,000** / 0,448 |
| Simuletic Scene6: mAP50 / mAP50-95 | 0,491 / 0,119 | **0,616** / **0,210** |
| Open Images test (320 imágenes reales): precisión / recall | 0,010 / 0,003 | **0,829** / **0,582** |
| Open Images test: mAP50 / mAP50-95 | 0,001 / 0,000 | **0,661** / **0,443** |
| 500 negativos difíciles (teléfono, paraguas, herramientas…): imágenes con falsa alarma | 98 (19,6 %) | **19 (3,8 %)** |
| Video del pasillo: frames con el rifle detectado (de 106 con el rifle visible) | 18 (17 %), solo al final | **73 (69 %)**, desde el frame 40 |
| Video del pasillo: falsas alarmas sin rifle visible (frames 0–38) | 0 | **0** |

`person` en Simuletic Scene6 sube de mAP50 0,856 a 0,938.

**Lectura:**

- v1 no generalizaba nada fuera de lo sintético: 0,1 % de AP en imágenes reales y
  falsas alarmas en una de cada cinco imágenes con teléfono o herramienta.
- v2 detecta armas reales, cuadruplica los frames detectados en el video real y reduce
  las falsas alarmas en negativos difíciles a una quinta parte.
- En la escena sintética lejana (Scene6) el recall no mejora (0,45): las armas de pocos
  píxeles siguen siendo el límite. Para eso hace falta metraje CCTV real, no fotos.

**Límites que siguen:**

- Las fotos de Open Images suelen mostrar el arma grande y en primer plano, a diferencia
  del CCTV real.
- 19 de los 500 negativos todavía generan alarma.
- El demo de 30 s con metraje real sigue pendiente.

## v3: aumentos de dominio + detección en dos etapas (2026-10-01)

Tres mejoras sin datos nuevos, comparadas con el **mismo evaluador**
(`scripts/firearm.py compare`, `src/firearm/evaluate.py`). El umbral se calibra en
validación por F1 máximo, nunca en test.

1. **Aumentos de dominio** (`src/firearm/augment.py`, modelo v3):
   - 1954 copias degradadas estilo CCTV: baja resolución, JPEG de calidad 15–50,
     desenfoque, poca luz con ruido y gris;
   - 1999 composiciones *copy-paste*: personas armadas de Open Images reducidas a
     50–220 px y pegadas con borde difuminado sobre frames reales de MOT16 (train) y
     escenas de train de Simuletic;
   - rango de escala `scale=0.9`;
   - 60 épocas, 39 min.
2. **Dos etapas** (`src/firearm/twostage.py`): personas con YOLOv8n COCO a 1280 y
   detector de armas sobre cada persona recortada y ampliada a 384 px. Se fusiona con
   la pasada sobre el frame completo por NMS.
3. **Umbral calibrado** en validación en vez del 0,35 fijo.

| Variante | Umbral (val) | Scene6 CCTV AP50 | Scene6 P / R | Open Images AP50 | OI P / R | Negativos con falsa alarma | Video: frames con el rifle |
|---|---|---|---|---|---|---|---|
| v2 | 0,475 | 0,617 | 0,67 / 0,13 | **0,608** | 0,93 / 0,53 | **2,0 %** | **51** |
| v2, dos etapas | 0,55 | 0,723 | 0,86 / 0,40 | 0,578 | 0,82 / 0,54 | 2,4 % | 44 |
| **v3** | 0,475 | **0,926** | **1,00 / 0,67** | 0,605 | 0,92 / 0,50 | 3,0 % | 44, desde el frame 37 |
| v3, dos etapas | 0,525 | 0,773 | 0,69 / 0,60 | 0,572 | 0,82 / 0,50 | 3,0 % | 39 |

- El conteo del video es sobre frames con el rifle visible.
- El frame 37–38, que en un principio parecía una falsa alarma, ya muestra el rifle
  en la mano. v3 lo detecta antes que v2.

Con el umbral anterior de 0,35, v3 da en Scene6 precisión / recall 1,00 / **0,87**,
en Open Images 0,91 / 0,55, un 5,8 % de falsas alarmas en negativos y 64 frames con
el rifle en el video.

**Lectura:**

- Los aumentos de dominio son la mejora clave para CCTV lejano. En Scene6 el AP50 pasa
  de 0,617 a 0,926 y el recall al umbral de 0,13 a 0,67. En validación solo-CCTV
  (Scene5, 13 armas) el AP50 sube de 0,32 a 0,67. En fotos de Open Images no cambia.
- El coste es algo más de falsas alarmas en negativos (2,0 % a 3,0 %) y algo menos de
  recall por frame en el video a igual umbral.
- Las **dos etapas** ayudan a un modelo que no ve armas pequeñas (v2), pero con v3
  empeoran: añaden detecciones dudosas en los recortes y hacen una inferencia extra
  por persona. Quedan disponibles (`TwoStageDetector`) pero **no** son la opción por
  defecto.
- Scene6 es sintética. Los fondos del *copy-paste* son de train y ninguno es Scene6,
  pero el estilo sintético se parece. La confirmación definitiva requiere CCTV real.

**Decisión:** el modelo por defecto pasa a **v3 en una etapa**, con `conf=0.475`
calibrado. Si en el sitio se prefiere recall a falsas alarmas, `conf=0.35`. El umbral
debe recalibrarse con video real del sitio.

## Conclusiones y siguientes pasos

1. El flujo completo funciona en la laptop: datos → entrenamiento → evaluación → video anotado con métricas.
2. El modelo **no sirve todavía como detector fiable**: tiene recall bajo en video real y en objetos pequeños.
3. Para mejorarlo hacen falta datos reales: conseguir CCTV-Gun (pedírselo a los autores) u otro dataset real con licencia clara, más hard negatives (teléfonos, herramientas, paraguas).
4. Pendiente en la Jetson Orin Nano 8 GB: exportar `best.pt` a TensorRT FP16 en la propia Jetson y repetir `annotate` midiendo FPS, p50/p90 y memoria sin OOM.
