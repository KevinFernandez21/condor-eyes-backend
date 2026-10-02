# Informe: verificación facial de personal autorizado (issue #10)

> **Hardware y runtime de estas mediciones:** laptop con Intel Core Ultra 9 275HX (24 hilos), 32 GB de RAM, GPU dedicada NVIDIA GeForce RTX 5080 Laptop (16 GB) y GPU integrada Intel Graphics, Windows 11, Python 3.11.16. Runtime: OpenCV 5.0.0 DNN en CPU, **FP32** (YuNet y SFace ONNX). La política vigente pasa a OpenVINO FP16 (equivalencia medida: coseno ≥ 0,9998 frente a FP32); ver la política de precisión del presupuesto edge. **No** es la Jetson Orin Nano ni la laptop i7 de 12.ª gen. del equipo: los FPS y latencias de aquí no se transfieren. El presupuesto común sin CUDA está en [`docs/edge-budget.md`](../edge-budget.md) (issue #23).

Corrida del 2026-10-01 en la laptop, con OpenCV DNN en CPU.
Comando: `uv run python scripts/faceid.py benchmark`.

## Datos y protocolo

- **DigiFace-1M** (Microsoft): caras **100 % sintéticas**, licencia de investigación no
  comercial. Se usaron 400 identidades × 24 imágenes de la parte P1, bajadas por HTTP
  Range (`faceid.py download`). Las imágenes son de 112×112 y varían en pose, iluminación,
  accesorios y expresión.
- **Sujetos disjuntos:** los ids 0–199 son validación y los 200–399, test. Un `assert`
  impide la fuga de identidad.
- **En cada split:**
  - 100 personas **enroladas** con 5 imágenes; sus otras 19 imágenes son sondas genuinas.
  - 100 personas **desconocidas** que nunca se enrolan; sus 19 imágenes son sondas impostoras.
  - En total: 1900 sondas genuinas y 1900 desconocidas en test.
- **Métricas** (identificación abierta 1:N):
  - **FAR:** desconocidos aceptados + enrolados aceptados como **otra** persona, sobre todas las sondas.
  - **FRR:** sondas genuinas que no terminan en `match` con su propia identidad. Incluye inconcluso y sin cara.
- **Calibración solo en validación:**
  - nitidez mínima = percentil 5 de las caras limpias de val, que dio 4,4;
  - umbral de similitud = el menor con FAR ≤ 0,1 % en val;
  - margen = 0,03.

## Selección de modelo (test, caras sin perturbar)

| Embedding | Umbral (val) | **FAR** | **FRR** | Inconcluso | Desconocido (genuino) | Desconocidos rechazados | Latencia detección + embedding p50 / p90 |
|---|---|---|---|---|---|---|---|
| **SFace FP32** | 0,65 | **0,00 %** | **42,8 %** | 24,8 % | 17,7 % | 76,1 % | 5,7 / 6,7 ms |
| SFace INT8 | 0,62 | 0,18 % | 40,0 % | 24,8 % | 14,8 % | 75,8 % | 13,2 / 14,6 ms |
| MobileNetV3 ImageNet (no facial) | 0,90 | 0,29 % | 92,6 % | 35,3 % | 56,6 % | 67,3 % | 24,6 / 28,7 ms |

- El 24 % restante de desconocidos que no se rechazan sale inconcluso por calidad, no
  como `match`.
- **Elegido: SFace FP32.** Es el único sin falsas aceptaciones en test y es el más rápido
  en CPU. INT8 ahorra memoria (9,4 MB frente a 37 MB) pero acepta algún impostor y en
  CPU x86 es más lento. Se puede reconsiderar en la Jetson.
- El baseline genérico confirma que hace falta un modelo facial: rechaza al 93 % del
  personal.

Curva de SFace FP32 en test (FAR / FRR):

| Umbral | 0,50 | 0,55 | 0,60 | **0,65** | 0,70 | 0,75 |
|---|---|---|---|---|---|---|
| FAR | 7,4 % | 2,9 % | 0,66 % | **0 %** | 0 % | 0 % |
| FRR | 30,8 % | 32,7 % | 36,2 % | **42,8 %** | 51,5 % | 64,7 % |

El FRR mínimo es 29 % aunque el umbral baje mucho. Ese suelo lo ponen las sondas
inconclusas por calidad, sobre todo los perfiles.

## Comportamiento ante condiciones adversas (SFace FP32, umbral 0,65)

| Condición | FAR | FRR | Inconcluso | Sin cara | Genuino tomado por desconocido |
|---|---|---|---|---|---|
| Original | 0 % | 42,8 % | 24,8 % | 0,3 % | 17,7 % |
| Desenfoque (gaussiano σ = 3) | 0 % | 99,6 % | **94,8 %** | 3,5 % | 1,3 % |
| Poca luz (gamma 2,2 al 50 %) | 0 % | 90,7 % | 66,1 % | **19,2 %** | 5,4 % |
| Oclusión de ojos (gafas) | 0 % | 96,9 % | 24,2 % | 1,7 % | **71,0 %** |
| Oclusión de boca (mascarilla) | 0 % | 74,1 % | 19,8 % | 9,1 % | 45,1 % |

| Pose (giro estimado por puntos) | Sondas | FRR | Inconcluso |
|---|---|---|---|
| Frontal (< 0,15) | 810 | **25,3 %** | 4,4 % |
| Media (0,15–0,35) | 677 | 28,8 % | 4,0 % |
| Perfil (> 0,35) | 408 | 100 % | 100 %, por diseño (`max_yaw`) |

Dos caras distintas en la misma imagen dan `multiple_faces` en el **98 %** de los casos.
En el 2 % restante YuNet detectó solo una cara y se verificó esa.

## Modos de fallo conocidos

1. **Perfiles:** se devuelven como inconclusos (`pose`). Es una decisión de seguridad:
   SFace es poco fiable con caras giradas.
2. **Desenfoque y poca luz:** casi todo queda inconcluso o sin cara. No hay falsas
   aceptaciones, pero el personal no se reconoce. En las cámaras reales hace falta buena
   iluminación y obturación rápida en los puntos de verificación.
3. **Gafas oscuras:** el 71 % del personal sale como **desconocido**, no como inconcluso,
   porque la cara es nítida pero el embedding cambia mucho. Un detector de oclusión haría
   que esos casos salieran como inconclusos.
4. **Mascarilla:** el 45 % sale como desconocido. Mismo remedio.
5. **Dominio sintético:** DigiFace no representa la cámara real (resolución, compresión,
   ángulo cenital). Hay que repetir la calibración con personal que consienta, en las
   cámaras del sitio, y medir FAR y FRR por grupo demográfico.

## Decisión

- **SFace FP32 + YuNet**, umbral 0,65, margen 0,03, nitidez mínima 4,4 y giro máximo
  0,35, todo calibrado en val. Hay que recalibrar en el sitio.
- La salida es **evidencia para el operador** (`requires_operator: true`). Con un FRR del
  25–43 %, el sistema **no puede** usarse como control de acceso automático, y no debe.
- Pendiente para la Jetson: medir latencia y memoria con OpenCV DNN (CPU o CUDA)
  junto al detector compartido.
