# Fusión de evidencia (issue #15)

Correlaciona pistas de video, verificación facial, re-ID, ubicación de tags, permisos y
reglas de zona, y entrega al operador un **registro de decisión**. El módulo es Python
puro (`src/fusion/`), sin reloj propio ni E/S: `FusionEngine.evaluate(evidencia, now)`.

**El operador humano es siempre la autoridad final.** Ningún resultado equivale a
autorizar o denegar acceso: todo registro lleva `requires_operator: true` y
`DecisionRecord` rechaza construirse con `False`.

## Entradas tipadas (`fusion.models`)

| Tipo | Origen | Campos relevantes |
|---|---|---|
| `TrackObservation` | tracker | `track_ref` (único entre cámaras), `stream_id`, `zone_id`, `confidence` |
| `IdentityEvidence` | verificación facial (#10) | `status` (`match`, `unknown`, `inconclusive`, `no_face`, `multiple_faces`, `invalid_input`), `person_id`, `score` |
| `ReidLink` | re-ID (#11) | `status` (`linked`, `inconclusive`, `no_candidate`), pistas origen/candidata, `confidence` |
| `LocationEvidence` | localización de tags (#16) | `person_id` ya resuelto (sin ID de tag), `zone_id`, `confidence`, `valid` |
| `ZoneRule` | configuración | `restricted` |
| `PermissionRepository` | configuración | `allowed_zones(person_id)`; `None` = sin registro |

Cada evidencia trae `evidence_id` (el ID del mensaje de origen) y `observed_at` con zona
horaria. `identity_from_envelope` y `reid_from_envelope` convierten sobres del bus y
descartan todo salvo los campos tipados (por ejemplo, `embedding`).

## Salida

`DecisionRecord` (`to_envelope()` → `MetadataEnvelope`, `source="fusion"`, publicado en
`Topic.EVENTS` con `publish_decisions`):

- `outcome`: `corroborated`, `alert`, `inconclusive`, `not_restricted`.
- `confidence`: **mínimo** de las confianzas de la evidencia que contribuye.
- `reason_codes`, en orden estable.
- `evidence`: lista de `EvidenceRef` con `evidence_id`, `kind`, `role`
  (`supports`/`conflicts`/`stale`/`context`), `observed_at` **de la fuente** y confianza.
- `decision_id`: hash determinista de pista, persona, zona, instante y evidencia.

### Trazabilidad y privacidad

Una alerta se traza hasta su origen por `decision_id` → `evidence[].evidence_id` (ID del
mensaje original, con `stream_id` y marca de tiempo). El registro no contiene
embeddings, plantillas, imágenes, cajas ni IDs de tag; solo el `person_id` ya
seudonimizado que entregan identidad y localización.

## Reglas de decisión

Solo `corroborated` exige **todo** lo siguiente, vigente y por encima de su umbral:
pista, rostro con `match`, tag de esa misma persona en la zona de la pista, permiso para
esa zona y confianza combinada >= `min_decision_confidence`. Cualquier evidencia
ausente, caducada, inválida o de baja confianza deja el resultado en `inconclusive` o
`alert`; nunca en `corroborated`.

| Código | Cuándo | Resultado |
|---|---|---|
| `person_without_tag` | Persona en zona restringida y sin tag vigente que la respalde | `alert` |
| `tag_without_person` | Tag vigente en zona restringida sin pista compatible (decisión sin `track_ref`) | `inconclusive` |
| `hidden_face` | Verificación `no_face` | `inconclusive` |
| `stale_sensor` | La lectura del tag caducó o quedó fuera de la ventana de correlación | `inconclusive` |
| `multiple_nearby_people` | Varias personas/tags en la zona y no hay identidad que desambigüe | `inconclusive` |
| `unknown_face` | Rostro de buena calidad no enrolado | `alert` |
| `identity_tag_mismatch` | El tag de la persona identificada está en otra zona | `alert` |
| `zone_not_permitted` | La persona no tiene permiso para la zona | `alert` |
| `no_permission_record` | No existe registro de permisos de la persona | `inconclusive` |
| `face_inconclusive`, `identity_missing`, `stale_identity`, `stale_vision`, `location_invalid`, `clock_skew`, `low_confidence`, `zone_unknown` | Evidencia no utilizable o ambigua | `inconclusive` |

Con rostro oculto no se atribuye un tag a una persona aunque haya uno solo en la zona: se
devuelve `inconclusive` con el tag como contexto para el operador. Un re-ID `linked`
vigente y sobre `min_reid_confidence` puede heredar la identidad de una pista con
`match` en otra cámara; nunca contradice un `unknown_face`.

## Determinismo

La evidencia se ordena por `(observed_at, evidence_id)`: el mismo conjunto produce el
mismo resultado, y los tests cubren repetición y barajado. `now` es un parámetro.

## Configuración de umbrales (`configs/fusion.toml`)

Sección `[fusion]`, cargada con `load_fusion_config("configs/fusion.toml")`. Las claves
desconocidas o fuera de rango fallan con un `ValueError`. Los valores son un punto de
partida y deben recalibrarse con datos del sitio.

| Clave | Defecto | Efecto |
|---|---|---|
| `time_window_s` | 5.0 | Diferencia máxima entre pista y rostro/tag para correlacionarlos; fuera de ella la evidencia cuenta como caducada |
| `track_max_age_s` | 3.0 | Antigüedad máxima de la pista respecto a `now` (`stale_vision`) |
| `identity_max_age_s` | 10.0 | Ídem para el rostro (`stale_identity`) |
| `location_max_age_s` | 15.0 | Ídem para el tag (`stale_sensor`); debe cubrir el intervalo de emisión de los nodos ESP32 |
| `reid_max_age_s` | 30.0 | Ídem para asociaciones de re-ID |
| `max_clock_skew_s` | 2.0 | Una marca posterior a `now` + este valor es `clock_skew` |
| `min_track_confidence` | 0.5 | Por debajo, la pista no se evalúa (`low_confidence`) |
| `min_identity_score` | 0.65 | Mínimo de un `match` para usarlo; alineado con el umbral del issue #10 |
| `min_location_confidence` | 0.6 | Mínimo de la zona estimada del tag |
| `min_reid_confidence` | 0.7 | Mínimo para heredar identidad por re-ID |
| `min_decision_confidence` | 0.6 | Mínimo del conjunto para `corroborated` |

Subir un umbral o acortar una ventana hace al sistema más conservador (más
`inconclusive`/`alert`); nunca más permisivo.

## Fuera de alcance

Entrenar modelos faciales o de re-ID, abrir puertas, notificar a autoridades y la
interfaz del operador.
