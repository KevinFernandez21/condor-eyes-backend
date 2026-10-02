# Informe: SFace vs Gemini Embedding 2 vs CNN genérica (issue #27)

## Protocolo

- Datos: subconjunto **pequeño** de DigiFace-1M (sintético): 60 identidades, 12 imágenes
  cada una. 30 sujetos para validación y 30 para test, **disjuntos**. En cada split, la mitad
  se enrola (5 imágenes) y la otra mitad son desconocidos; 7 sondas por sujeto.
- Mismo detector (YuNet), misma puerta de calidad y misma decisión de conjunto abierto para
  todos los embedders. El umbral se calibra en validación con FAR objetivo 1 % (el
  subconjunto es pequeño; con 105+105 sondas no se puede resolver un 0,1 %).
- Hardware: portátil x86_64 (AMD Zen 3), Windows 11, **CPU** (OpenCV DNN / torch). GPU no usada.
- Comando: `uv run python scripts/faceid.py compare-embedders --subjects 60 --per-subject 12 --target-far 0.01`.

## Resultados (test, caras sin perturbar)

| Embedder | Dim | Umbral (val) | FAR | FRR | Latencia p50 (embedding) | Coste |
|---|---|---|---|---|---|---|
| SFace FP32 | 128 | 0,52 | 0,48 % | 29,5 % | 4,4 ms | 0 |
| MobileNetV3 ImageNet (baseline) | 576 | 0,90 (*) | 0 % | 83,8 % | 5,0 ms | 0 |
| Gemini Embedding 2 | 768 | - | - | - | - | **no ejecutado** |

(*) Con la CNN genérica no se alcanzó el FAR objetivo en el rango de umbrales explorado (máx. 0,90): sus
vectores de caras distintas son demasiado parecidos. El FAR de 0 % se consigue a costa de rechazar
casi todo.

Gemini: **no ejecutado**, porque `GEMINI_API_KEY` no estaba definida en esta máquina. No
hay ningún número de Gemini en este informe. Coste estimado de una ejecución como la anterior:
~460 imágenes x 0,00012 USD = ~0,06 USD en nivel de pago (gratis, con las condiciones de datos
descritas en `docs/face-verification.md`).

El FRR de SFace es coherente con el informe de #10: lo domina el suelo de calidad
(`inconclusive`/`no_face` en caras sintéticas pequeñas), no la comparación.

## Decisión

Pendiente para Gemini. SFace local sigue siendo el modelo por defecto. Para completar:
definir `GEMINI_API_KEY` y repetir el comando (el informe se regenera en `reports/`, fuera de Git).
