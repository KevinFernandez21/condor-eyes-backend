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

## Búsqueda por embeddings y comparación con Gemini Embedding 2 (issue #27)

Extiende el módulo anterior sin crear un contrato paralelo: sigue usando `VerifyStatus`,
`VerificationResult`, las reglas de `EnrollmentStore` y `MetadataEnvelope`.

| Pieza | Archivo |
|---|---|
| Protocolo `Embedder` + SFace, CNN genérica y Gemini | `src/faceid/embedders.py` |
| Índice vectorial (numpy + SQLite), coseno top-k y decisión de conjunto abierto | `src/faceid/vectorstore.py` |
| Rol `identity`: recorte de cara a `MetadataEnvelope` | `src/faceid/identity.py` |
| Enrolamiento por webcam | `scripts/face_enroll.py` (lógica en `src/faceid/enroll*.py`) |
| Benchmark SFace vs Gemini vs CNN | `uv run python scripts/faceid.py compare-embedders` |

### Índice vectorial

- Guarda **solo vectores**, referencia de persona, referencia de consentimiento y caducidad.
  Sin imágenes. `delete`, `purge_expired` y auditoría JSONL sin datos biométricos, como en #10.
- **Un índice por embedder** (`data/faceid/index_<model_id>.sqlite`): los vectores de modelos
  distintos no son comparables. Abrir un índice con otro `model_id` falla.
- Se guardan todos los vectores de cada persona; la búsqueda devuelve las top-k **personas**
  (mejor vector de cada una).
- Decisión abierta: `unknown` si nadie alcanza el umbral; `inconclusive` si el mejor y el
  segundo están a menos de `margin`; `match` en otro caso. Los umbrales **no se transfieren**
  entre embedders: se calibran con validación para cada uno.

### Rol `identity`

`IdentityHandler.identify(recorte) -> MetadataEnvelope` (`source="identity"`). El recorte entra
por el lado del pipeline, nunca por el bus. El payload lleva `status`, `person_id`, `score`,
`threshold`, `model_id`, `observed_at` (ISO-8601 como texto), `evidence` y
`requires_operator: true`. Es JSON estricto: solo primitivas finitas (sin numpy, `datetime`
ni NaN; `json_safe` lo garantiza). Ni imágenes ni embeddings viajan en el envelope.
Conectarlo al runtime de agentes queda como seguimiento (no se toca `bus/hub.py` ni `agents/`).

### Gemini Embedding 2: reglas de uso (nube)

- Modelo `gemini-embedding-2` (GA abril 2026), SDK `google-genai`, importado de forma diferida.
  Se usa a 768 dimensiones (rango admitido 128-3072) y se re-normaliza (L2) en local.
  Precio de pago: 0,00012 USD por imagen; hay nivel gratuito.
- **Clave solo en la variable de entorno `GEMINI_API_KEY`.** Nunca se escribe en archivos, ni
  se registra; los mensajes de error la redactan.
- **Guardia de consentimiento de nube** (`cloud_consent=True`), distinta del consentimiento
  local. Sin ella, `GeminiEmbedder` lanza `CloudConsentError` antes de codificar o enviar nada;
  el índice de Gemini rechaza enrolar sin ella; el `IdentityHandler` rechaza un embedder de nube
  salvo `allow_cloud=True`, porque cada **sonda** también enviaría la cara de un transeúnte.
- **Nivel gratuito: Google puede usar el contenido enviado para mejorar sus productos**; solo
  el nivel de pago lo excluye. Por eso solo se envían caras de personas que hayan firmado un
  consentimiento específico para esta prueba (inicialmente Kevin) y caras sintéticas
  (DigiFace-1M). El uso en producción de un embedding en la nube para biometría exige nivel de
  pago o se rechaza; la ruta local sigue siendo la predeterminada.
- Límites de uso: lotes de 8 imágenes (un `Content` por imagen, para obtener un vector por
  imagen) y reintentos con espera exponencial ante 429.

### Enrolar con la webcam

```bash
# Prueba sin cámara, modelos ni red; no escribe nada
uv run python scripts/face_enroll.py --dry-run --person-id EMP-014 --consent-ref CONS-2026-031

# Local (SFace): captura 5 fotos, guarda solo vectores
uv run python scripts/face_enroll.py --person-id EMP-014 --consent-ref CONS-2026-031

# Nube (solo con consentimiento específico y GEMINI_API_KEY en el entorno)
uv run python scripts/face_enroll.py --embedder gemini --cloud-consent --person-id EMP-014 --consent-ref CONS-2026-031
```

Los fotogramas se procesan en memoria y se descartan; no se escriben a disco.

### Resultados de la comparación

Ver [`docs/reports/face-embedding-search.md`](reports/face-embedding-search.md). La decisión
sobre Gemini se mantiene **pendiente** hasta ejecutar el benchmark con clave: solo se adopta
si iguala o supera a SFace en identidad; si no, se documenta como rechazado con números.
