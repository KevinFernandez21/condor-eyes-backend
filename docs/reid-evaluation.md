# Re-identificación entre cámaras (issue #11)

Objetivo: cuando el tracker local pierde a una persona al pasar a otra cámara sin
solape, proponer **qué track de otra cámara es la misma persona** o devolver
**inconcluso**. Nunca se fuerza un enlace.

Código en `src/reid/`, CLI en `scripts/reid.py` y resultados en
[`docs/reports/reid-benchmark.md`](reports/reid-benchmark.md).

## Contrato

Entrada, `TrackDescriptor`:

- `stream_id`, `track_id`;
- `t_first`, `t_last`;
- `embedding`: vector de apariencia;
- `zone`, opcional.

El tracker de cada cámara publica un descriptor por track local. El recorte de imagen
**se queda en el plano de video**: solo viaja el vector.

Salida, `AssociationResult` → `to_envelope()` → `MetadataEnvelope`:

| Campo | Contenido |
|---|---|
| `source_stream`, `source_track` | Track que se quiere continuar |
| `candidate_stream`, `candidate_track` | Mejor candidato, o `null` |
| `status` | `linked`, `inconclusive` (similitud baja o ambigüedad) o `no_candidate` |
| `confidence` | 0–1 |
| `evidence` | Similitud, segundo mejor, margen, Δt, zona del candidato, candidatos considerados y rechazados por motivo |

El embedding no se incluye en el payload.

`CrossCameraAssociator`:

- `upsert`: alta o actualización. El mismo `(cámara, track)` nunca se duplica; su
  embedding se promedia.
- `associate`:
  - descarta los candidatos de la **misma cámara**, los que **se solapan en el tiempo**
    y los **obsoletos** (`max_age_s`);
  - si hay tabla de transiciones, también los que no tienen una **transición permitida**
    o caen fuera de su **tiempo de tránsito**.
  - Con lo que queda:
    - `no_candidate` si no queda ningún candidato;
    - `inconclusive` si la similitud es menor que `threshold` o el margen sobre el
      segundo track distinto es menor que `margin`;
    - `linked` en otro caso.
- `prune(now)`: aplica la retención y borra los descriptores viejos.

La decisión final de identidad, si la hay, la toma el operador. El enlace es evidencia
para el tracking continuo, no una identificación.

## Métodos comparados

| Método | Tipo | Licencia |
|---|---|---|
| `color-hsv` | **Baseline no biométrico**: histograma HSV (Hellinger) de torso y piernas; descarta el 15 % superior del recorte (cabeza). En el sitio se combina con la tabla de zonas y tiempos de tránsito | Código propio |
| `mobilenet_v3_small-imagenet`, `resnet18-imagenet` | CNN de ImageNet sin entrenar para re-ID | torchvision (BSD-3); pesos entrenados en ImageNet |
| `*-reid` | La misma CNN con *fine-tuning* re-ID: entropía cruzada con *label smoothing* + triplete *batch-hard*, BNNeck, 40 épocas, 256×128 | Pesos derivados de Market-1501: **solo investigación** |

## Datos y splits

**Market-1501** (v15.09.15): 6 cámaras, 1501 identidades y recortes del detector DPM.
Su licencia es de uso **solo para investigación**; se usó así, según lo acordado, y los
pesos derivados heredan la restricción.

| Split | Identidades | Uso |
|---|---|---|
| train | 676 (de `bounding_box_train`) | Fine-tuning |
| val | 75 separadas de train (semilla 0) | Calibrar `threshold` y `margin` |
| test | 750 oficiales (`query` / `bounding_box_test`, con distractores) | Solo se reporta |

Las identidades de train, val y test no se solapan; un test lo comprueba.

El set cubre cambios de cámara, oclusión parcial, recortes desalineados del detector y
ropa parecida. **No** cubre escenas muy concurridas ni tiempos reales entre cámaras: los
números de frame de Market no están sincronizados entre cámaras. Por eso el filtro de
tiempo y zona se valida con tests unitarios y no con el benchmark.

## Métricas

- **Recuperación** (protocolo Market): rank-1, rank-5 y mAP, excluyendo la misma
  identidad en la misma cámara.
- **Asociación con estado inconcluso:** la galería se agrupa en tracks
  (identidad × cámara; cada distractor es un track). Cada consulta se asocia contra las
  otras cámaras en dos escenarios:
  - **presente** (su identidad está): se cuentan enlaces correctos, **falsos enlaces**
    (persona equivocada) y **enlaces perdidos** (inconcluso).
  - **ausente** (se quitan sus tracks): cualquier enlace es **falso**.
- **Calibración en val:** se maximizan los enlaces correctos con una tasa de falsos
  enlaces ≤ 1 %. Un falso enlace es peor que uno perdido, porque mezcla el historial de
  dos personas.
- **Coste:** latencia por recorte (batch 1, p50/p90) y por lote de 16; parámetros;
  tamaño de los pesos en FP16; pico de memoria CUDA.

```bash
uv run python scripts/reid.py train --arch resnet18
uv run python scripts/reid.py train --arch mobilenet_v3_small
uv run python scripts/reid.py benchmark
```

## Privacidad y retención

Un embedding de apariencia **es dato personal**: permite reconocer a alguien mientras
lleve la misma ropa. Reglas para el despliegue:

1. **Sin imágenes:** solo se guardan vectores en memoria, nunca recortes. El agente
   `storage` guarda clips de eventos con su propia política, no por re-ID.
2. **Retención corta:** `max_age_s` (600 s por defecto) y `prune()` periódico. No hay
   persistencia en disco ni enlaces entre días.
3. **Alcance local:** solo entre las cámaras del sitio. Nada de búsquedas en bases
   externas ni identificación masiva (fuera de alcance del issue).
4. **Mínimo necesario:** el baseline de color descarta la cabeza, y la CNN no se
   entrena ni se ajusta con caras.
5. **Acceso y auditoría:** los enlaces se publican como evidencia con su similitud y
   motivo, y cualquier uso fuera del tracking continuo (por ejemplo, una búsqueda
   retroactiva) debe registrarse y aprobarlo un operador.
6. **Sesgos:** Market-1501 es un campus universitario. La exactitud puede variar con la
   ropa, la iluminación y la demografía del sitio, así que hay que validar con datos
   locales y consentidos antes de usarlo.
