# Verificación facial de personal autorizado (issue #10)

Objetivo: que el agente de identidad distinga a una **persona enrolada con
consentimiento** de una **desconocida**, entregando **evidencia como metadata**. El
modelo nunca decide acceso; la decisión final es de un operador.

Código en `src/faceid/`, CLI en `scripts/faceid.py` e informe en
[`docs/reports/face-verification.md`](reports/face-verification.md).

## Selección de modelo

| Componente | Modelo | Licencia | Tamaño | Notas de despliegue |
|---|---|---|---|---|
| Detector | **YuNet** 2023mar (OpenCV Zoo) | MIT | 0,23 MB | Da caja, 5 puntos y score; corre en `cv2.FaceDetectorYN` sin dependencias nuevas |
| Embedding (elegido) | **SFace** 2021dec FP32 (OpenCV Zoo) | Apache-2.0 | 37 MB | MobileFaceNet; 128-d; `cv2.FaceRecognizerSF.alignCrop` usa los 5 puntos de YuNet |
| Embedding (edge) | SFace INT8 (OpenCV Zoo) | Apache-2.0 | 9,4 MB | Cuantizado; menos memoria para la Jetson |
| Baseline | MobileNetV3-small ImageNet | BSD-3 (torchvision) | 3,7 MB | **No** es un modelo facial; sirve para medir cuánto aporta uno específico |

Se descartaron:

- Modelos de InsightFace/ArcFace: los pesos públicos son solo para investigación no comercial.
- Cualquier modelo que exija servicios en la nube.

La entrada esperada es una cara de al menos 40 px. Los recortes pequeños se agrandan a
224 px antes de YuNet.

## Contrato

`Verifier.verify(imagen BGR)` → `VerificationResult` → `to_envelope()` → `MetadataEnvelope`
con `source="identity"`:

| `status` | Cuándo |
|---|---|
| `match` | Similitud con la mejor plantilla ≥ umbral **y** margen ≥ `margin` sobre la segunda |
| `unknown` | Cara de buena calidad que no alcanza el umbral con nadie |
| `inconclusive` | Calidad insuficiente (cara pequeña, score de detección bajo, desenfoque, giro) o dos enrolados demasiado parecidos |
| `no_face` / `multiple_faces` | Ninguna cara o más de una; nunca se elige una al azar |
| `invalid_input` | Imagen nula, corrupta, no BGR o demasiado pequeña |

El payload incluye `person_id`, `score`, `threshold` y `evidence` (calidad, segundo
score, número de enrolados). También lleva **`requires_operator: true`**. No hay ningún
campo de permitir o denegar acceso.

## Enrolamiento, verificación, borrado

```bash
uv run python scripts/faceid.py enroll --person-id EMP-014 --consent-ref CONS-2026-031 --retention-days 365 foto1.jpg foto2.jpg foto3.jpg
uv run python scripts/faceid.py verify captura.jpg
uv run python scripts/faceid.py delete --person-id EMP-014
uv run python scripts/faceid.py purge
uv run python scripts/faceid.py benchmark
```

## Privacidad

1. **Consentimiento obligatorio:** `enroll` exige `--consent-ref`, que referencia el
   consentimiento firmado y archivado fuera del sistema. Sin él, el enrolamiento falla.
   Solo se enrola personal autorizado que lo haya dado.
2. **Sin imágenes:** el almacén (`data/faceid/enrolled.json`, fuera de Git) guarda
   **solo la plantilla** (promedio de embeddings), el número de muestras, la referencia
   de consentimiento y las fechas. Las fotos de enrolamiento y las capturas de
   verificación no se guardan.
3. **Retención:** cada plantilla caduca (`--retention-days`, 365 por defecto). Las
   caducadas dejan de usarse de inmediato, y `purge` las borra.
   - **Revocación:** `delete` borra la plantilla al momento cuando la persona retira
     su consentimiento o deja la organización.
4. **Auditoría:** cada `enroll`, `verify`, `delete` y `purge_expired` se registra en
   `enrolled.audit.jsonl` con hora, acción, persona, actor y resultado, nunca con
   datos biométricos.
   - El acceso al almacén y al log debe limitarse al responsable de datos, con
     permisos del sistema de archivos y cifrado en reposo en el despliegue.
5. **Alcance:**
   - Solo verificación 1:N contra el personal enrolado del sitio.
   - Nada de identificación contra bases públicas, scraping de caras ni uso con
     personas que no hayan dado consentimiento.
   - Nada de denegar acceso automáticamente ni notificar a autoridades (fuera de
     alcance del issue).
6. **Datos de evaluación:** solo caras **100 % sintéticas** (DigiFace-1M), así que
   ninguna persona real necesita consentir. Antes de producción hay que validar con
   datos del personal que consienta, en las cámaras reales.
7. **Sesgo:** se debe medir FAR y FRR por grupos demográficos en la validación real.
   DigiFace no permite medirlo de forma fiable.
