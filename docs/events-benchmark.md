# Detección de eventos: merodeo, intrusión en zona y aglomeración (issue #12)

El agente `event` recibe **tracks** (metadata del tracker) y publica **eventos con
evidencia** en `Topic.EVENTS`. Nunca recibe ni envía frames.

Código en `src/events/`, configuración en `configs/events.toml` y CLI en
`scripts/events.py`. Los resultados y la recomendación están en
[`docs/reports/events-benchmark.md`](reports/events-benchmark.md).

## Contrato

Entrada, `TrackObservation`: `stream_id`, `track_id`, `t` (s), `x`, `y` y
`authorized`. `x`/`y` son el **punto de apoyo** (centro inferior de la caja)
normalizado a [0, 1]. `authorized` lo aporta el agente de identidad o fusión; el
detector de eventos solo lo usa para graduar la severidad.

Salida, `Event` → `Event.to_envelope()` → `MetadataEnvelope` con `source="event"` y un
payload solo de tipos primitivos:

| Campo | Contenido |
|---|---|
| `type` | `zone_intrusion`, `loitering` o `crowding` |
| `zone`, `track_ids` | Zona y tracks que lo disparan |
| `start_t`, `detect_t` | Inicio de la conducta e instante en que se reporta |
| `confidence` | 0–1 |
| `severity` | `alert`; `review` o `suppressed` si todos los tracks están autorizados |
| `authorized` | Si la persona o el grupo estaban autorizados |
| `evidence` | Tiempo dentro, desplazamiento neto, recorrido, puntos de entrada y último punto, conteo |

**Autorización:** con `authorized = "downgrade"` el evento baja a `review`; con
`"suppress"` queda como `suppressed`. En ambos casos **se publica con toda su
evidencia**. La persona autorizada nunca oculta el evento, solo cambia su severidad,
y la decisión final queda en el operador.

## Reglas (baseline)

| Evento | Regla | Parámetros (`[[zones]]`) |
|---|---|---|
| Intrusión | Track dentro de una zona `restricted` de forma continua ≥ `intrusion_min_s` | `restricted`, `intrusion_min_s` |
| Merodeo | Permanencia en la zona ≥ `loiter_dwell_s` **y** desplazamiento neto ≤ `loiter_max_disp` en los últimos `loiter_dwell_s` | `loiter_dwell_s`, `loiter_max_disp` |
| Aglomeración | ≥ `crowd_threshold` tracks a la vez dentro de la zona durante ≥ `crowd_dwell_s` | `crowd_threshold`, `crowd_dwell_s` |

Política global (`[policy]`):

- `max_gap_s`: un hueco mayor reinicia el track.
- `exit_grace_s`: histéresis de salida. Una salida breve no re-arma el evento.
- `authorized`: política de severidad para personas autorizadas.

Cada evento se emite una vez por episodio y se re-arma cuando el track sale de la zona.

Para una cámara real se dibujan los polígonos sobre un frame de referencia, con
coordenadas normalizadas, y se añade un bloque `[[zones]]` con su `stream_id`.

## Dataset de escenarios sintéticos

No existe un dataset público con licencia clara que tenga *tracks* y *eventos*
etiquetados para estas reglas y zonas. Por eso el benchmark genera escenarios
deterministas (`src/events/simulate.py`) con una cámara fija `cam-sim` de 180 s cada uno:

| Comportamiento | Rol |
|---|---|
| `transit` | Tránsito normal, nunca entra en la zona restringida |
| `slow_transit` | **Hard negative** de merodeo: cruza la plaza muy despacio |
| `short_wait` | **Hard negative**: se detiene entre 0,3 y 0,7 veces el umbral de merodeo |
| `border_stand` | **Hard negative**: parado a 0,012–0,025 (coordenadas normalizadas) del borde de la zona restringida |
| `group` | Grupo de 4 o 5 personas que se reúne entre 8 y 12 s (aglomeración o hard negative) |
| `loiter` | Merodeo: permanece entre 1,3 y 2,5 veces el umbral en un radio pequeño |
| `intrusion` | Entra en la zona restringida entre 1,5 y 10 s |
| `authorized_worker` | Persona autorizada estacionaria en la zona restringida o en la plaza |

Perturbaciones del tracker:

- ruido de posición;
- pérdida de detecciones;
- **oclusiones** (huecos de 0,5–3 s);
- **cambios de ID**;
- **vibración de cámara**.

**Reglas de etiquetado:** las etiquetas nunca salen del motor de reglas.

- Intrusión: la geometría real, sin ruido. `start_t` es el primer instante dentro de la zona.
- Merodeo: el guion. `start_t` es la entrada en la zona + `loiter_dwell_s`, el primer momento en que es reportable.
- Aglomeración: el conteo real. `start_t` es el inicio del conteo ≥ umbral + `crowd_dwell_s`.

**Splits por semilla, disjuntos:**

| Split | Semillas | Perturbaciones | Uso |
|---|---|---|---|
| train | 0–299 | normales | Entrenar el modelo aprendido |
| val | 1000–1099 | normales | Calibrar el umbral del modelo aprendido |
| **test** | 5000–5299 (15 h) | **más fuertes**: ruido ×1,75, oclusiones de hasta 3 s, el doble de cambios de ID y vibración | Solo se reporta; no se ajusta nada con él |

## Métricas

- **Emparejamiento** 1 a 1, mismo tipo, cámara y zona. La detección debe caer entre
  `start_t − 2 s` y `end_t + 5 s`. Los eventos de track exigen además compartir algún
  `track_id` con la etiqueta.
- **Precisión, recall, F1** por tipo de evento.
- **Falsas alarmas por hora:** eventos sin etiqueta / horas de escenario. Se reportan
  aparte de las **omisiones**.
- **Tiempo de detección (TTD):** `detect_t − start_t` (media, p50, p90).
- **Exactitud de severidad:** si la persona autorizada recibió `review` o `suppressed`.
- **Latencia** por observación (µs, p50/p90/p99) y **pico de memoria** de Python
  (`tracemalloc`).
- **Análisis de fallos:** cada falsa alarma u omisión se etiqueta con el
  comportamiento y las perturbaciones activas (oclusión, cambio de ID, vibración,
  grupo, autorizado).

## Comandos

```bash
uv run python scripts/events.py benchmark --learned --output reports/events_benchmark.json
uv run pytest tests/test_events.py
```

## Guía de anotación para video real

Cuando haya grabaciones del sitio, el mismo benchmark corre sobre datos reales:

1. **Tracks:** se exportan en formato MOTChallenge (`frame,id,x,y,w,h,...`, frame
   base 1). Pueden salir del tracker que se elija en ARCH-003 o de anotación manual
   en CVAT. Los IDs autorizados se pasan con `--authorized-ids`.
2. **Zonas:** se dibujan sobre un frame y se añaden a `configs/events.toml` con el
   `stream_id` de la cámara.
3. **Eventos:** se anotan en un CSV con columnas
   `stream_id,event_type,zone,start_s,end_s,track_ids,authorized`:
   - `event_type`: `zone_intrusion`, `loitering` o `crowding`.
   - `start_s`: el primer instante en que el evento es **reportable** según la
     política, por ejemplo entrada + umbral para merodeo. Así el TTD mide el retraso
     del sistema.
   - `end_s`: cuando termina la conducta.
   - `track_ids`: separados por `;`; vacío para aglomeración.
   - `authorized`: `sí` o `no`.
4. **Escenarios mínimos** que debe cubrir el held-out real: tránsito normal, merodeo,
   presencia autorizada (trabajadores parados), aglomeración, intrusión, y hard
   negatives (esperas cortas, gente junto al borde, grupos de paso, vibración).
5. Se anotan **horas completas de actividad normal**, no solo los clips con eventos.
   Sin eso no se pueden medir las falsas alarmas por hora.

```bash
uv run python scripts/events.py evaluate --tracks cam01_tracks.txt --ground-truth cam01_events.csv --stream-id cam01 --fps 15 --width 1920 --height 1080
```

## Privacidad

El detector solo maneja posiciones y IDs de track locales, sin imágenes ni rasgos
biométricos. La evidencia de un evento son coordenadas y tiempos. El clip asociado,
si existe, lo guarda el agente `storage` según su propia política de retención,
fuera del bus.
