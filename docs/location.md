# Localización de personal por zonas (BLE + LoRa opcional)

Estado: **solo lado Python, probado con simulador**. No se probó con hardware
real (ESP32-S3, gateway LoRa); los umbrales de `configs/location.toml` son
valores iniciales a calibrar en campo.

Fuera de alcance: inferencia de cámara, entrenamiento de rostros, control Pan-Tilt.

## Flujo

```
tag BLE ──anuncio──▶ nodo de zona ESP32-S3 ──JSON/binario──▶ LocationService
                                                               │ 1. parseo (payload.py)
                                                               │ 2. tag enrolado (repository.py)
                                                               │ 3. validación (validation.py)
                                                               │ 4. estimación (estimator.py)
                                                               ▼
                                          MetadataEnvelope ─▶ Topic.LOCATION
```

Módulo: `src/location/`. Los resultados salen solo por `MetadataEnvelope`
(`bus_adapter.py`); no hay frames ni IDs en claro en el bus.

## Contrato de payload (versión 1)

Cada nodo envía una observación por cada anuncio de tag que oye. El mismo
anuncio (misma `seq`) lo pueden reportar varios nodos.

### JSON (UTF-8, BLE/Wi-Fi/MQTT)

```json
{"v":1,"ch":"ble","tag":"A1B2C3D4E5F6","node":"N0007","rssi":-67,"ts":1767268800250,"seq":1234,"bat":87}
```

| Campo | Tipo | Oblig. | Descripción |
|-------|------|--------|-------------|
| `v` | int | sí | Versión del contrato; hoy `1`. |
| `ch` | string | no | `"ble"` (por defecto) o `"lora"`. |
| `tag` | string | sí | ID del tag: 12 dígitos hex (6 bytes), sin distinguir mayúsculas. |
| `node` | string | sí | ID del nodo, `[A-Za-z0-9_-]{1,32}`; debe existir en `[zones.*]`. |
| `rssi` | int | sí | RSSI en dBm medido por el nodo (esperado -127..0). |
| `ts` | int | sí | Epoch en **milisegundos UTC** según el reloj del nodo. |
| `seq` | int | sí | Contador del tag, 32 bits sin signo, +1 por anuncio, vuelve a 0 al desbordar. |
| `bat` | int | no | Batería del tag en % (0..100). Omitir si se desconoce. |

Se ignoran campos extra (compatibilidad hacia adelante). Cualquier otro
incumplimiento es `PayloadError` y el paquete se cuenta como `malformed`.

### Binario compacto (26 bytes, little-endian, LoRa/LoRaWAN)

| Offset | Tamaño | Campo | Notas |
|--------|--------|-------|-------|
| 0 | u8 | versión | `1` |
| 1 | u8 | flags | bits 0-1: canal (`0` BLE, `1` LoRa); resto reservado |
| 2 | 6 B | tag | bytes del ID |
| 8 | u16 | nodo | se mapea a `N` + 4 dígitos (`7` -> `N0007`) |
| 10 | u64 | ts | epoch ms UTC |
| 18 | u32 | seq | contador del tag |
| 22 | i8 | rssi | dBm |
| 23 | u8 | bat | % ; `0xFF` = desconocida |
| 24 | u16 | crc | CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) de los bytes 0..23 |

Struct de referencia para firmware: `"<BB6sHQIbBH"`. Vector de prueba del CRC:
`crc16(b"123456789") == 0x29B1`. En LoRa SF12 el payload máximo es 51 bytes;
26 B cabe con margen.

## Validación

Una observación se rechaza (y se cuenta en `LocationService.stats`) si es:

| Motivo | Condición |
|--------|-----------|
| `malformed` | No cumple el contrato (JSON/CRC/longitud/versión). |
| `unenrolled_tag` | El tag no está en el repositorio de personal (minimización de datos). |
| `unknown_node` | El nodo no está en `[zones.*]`. |
| `impossible_value` | RSSI fuera de `rssi_min_dbm..rssi_max_dbm` o batería fuera de 0..100. |
| `stale` | `recepción - ts > max_age_s`. |
| `clock_skew` | `ts - recepción > future_tolerance_s` (reloj del nodo adelantado). |
| `duplicate` | Misma `seq` ya vista para el par (tag, nodo). |
| `replay` | `seq` anterior a la última del par (aritmética circular de 32 bits). |
| `impossible_transition` | Lectura fuerte (`>= transition_min_rssi_dbm`) en zona no vecina a menos de `transition_min_s` de otra lectura fuerte del mismo tag. |

Un nodo cuyo dato se rechaza sigue contando como vivo (salvo `unknown_node`).

Limitaciones conocidas:

- El anti-replay es por contador, **sin autenticación criptográfica**: un
  atacante que adelante la `seq` podría bloquear a un tag. Mejora futura: HMAC
  por tag sobre el payload.
- Un único paquete con `seq` unos 2^31 por delante (o forjado) fija ese valor
  como "última secuencia" del par (tag, nodo): los paquetes legítimos siguientes
  se rechazan como `replay` hasta que expire el estado (`retention_s`). Es una
  limitación del contador sin autenticación por tag; la mitigación real es el
  HMAC por tag. Por eso `retention_s` debe ser >= `max_age_s`.
- Entradas hostiles: JSON de más de `MAX_JSON_BYTES` (1024), tipos distintos de
  `str`/`bytes` o anidamiento profundo se devuelven como `malformed`; el
  ingestor nunca lanza por datos externos.
- Un tag que se reinicia vuelve a `seq` baja y se rechaza como `replay` hasta
  que supere la última vista o pase `retention_s`. Mitigación futura: campo de
  época de arranque.
- Un nodo con reloj desviado más allá de la tolerancia queda descartado
  (visible en `stats`); sincronizar con NTP o con el gateway.

## Estimación de zona

Por tag: media móvil exponencial del RSSI por nodo dentro de `smoothing.window_s`;
la puntuación de una zona es el mejor RSSI suavizado de sus nodos; gana la mayor,
con histéresis (`hysteresis_db`) para evitar parpadeo.

`confidence = (0.6·fuerza + 0.4·margen) · frescura · muestras · penalización`,
en `[0, 1]`:

- fuerza: RSSI de la zona ganadora entre `rssi_floor_dbm` y `rssi_strong_dbm`;
- margen: ventaja sobre la segunda zona, hasta `margin_full_db`;
- frescura: baja hasta 0.5 al acercarse a `evidence_max_age_s`;
- muestras: `muestras / samples_full`, tope 1;
- penalización: `outage_penalty` si una zona vecina tiene todos sus nodos caídos
  (`degraded = true`).

Si no hay evidencia fresca el estado es `unknown` con `unknown_reason`:
`tag_missing` (nunca visto), `stale_evidence` (visto pero caducó) o
`node_outage` (los nodos de su última zona no dan señales de vida en
`node_timeout_s`). Los nodos reportan vida con cualquier paquete o con
`LocationService.heartbeat(node_id)`.

## Mensaje en el bus (`Topic.LOCATION` = `"location.estimates"`)

`MetadataEnvelope(source="location", payload=...)`, un sobre por tag enrolado:

```json
{
  "tag_ref": "tag-866dbbeac4f9", "person_ref": "person-3fa1c0d2b7e4",
  "status": "located", "zone_id": "lobby", "confidence": 0.93,
  "unknown_reason": null, "degraded": false, "authorized": true,
  "battery_pct": 90, "battery_low": false,
  "first_evidence_at": "2026-01-01T12:00:00+00:00",
  "last_evidence_at": "2026-01-01T12:00:04+00:00",
  "computed_at": "2026-01-01T12:00:04.500000+00:00",
  "evidence": [{"node_id": "N0001", "zone_id": "lobby", "smoothed_rssi_dbm": -59.0,
                "samples": 5, "first_seen_at": "...", "last_seen_at": "..."}]
}
```

`authorized` es `true/false` según `allowed_zones` del repositorio, o `null`
si el estado es `unknown`.

## Privacidad y control de acceso

- El mapeo tag -> persona vive detrás de `PersonnelRepository` (reemplazable:
  SQLite, LDAP, API de RR. HH.). Todo acceso exige un `Principal` con el alcance
  `personnel:read`; sin él se lanza `AccessDenied` y se audita el intento.
- `Principal` y sus alcances son control de acceso a nivel de contrato (el
  repositorio exige el alcance), **no autenticación**: no verifican quién es el
  llamador. La autenticación real (tokens, mTLS, identidad del servicio) debe
  hacerla la capa que construye el `Principal`.
- IDs de tag y de persona nunca van en claro a logs ni al bus: se usan
  seudónimos HMAC-SHA256 con clave secreta (`Pseudonymizer`). `repr` de
  `TagObservation` y `PersonnelRecord` oculta los datos personales.
- Clave de producción: debe inyectarse desde un secreto del despliegue (p. ej.
  Docker secret). `Pseudonymizer.random()` es solo para pruebas: las referencias
  cambian en cada arranque.
- Los tags no enrolados se descartan sin guardar estado.

## Cómo LoRaWAN extiende la cobertura (opcional)

BLE alcanza decenas de metros: sirve dentro de una zona. Donde no hay nodos BLE
cableados (patios, perímetros), un tag con radio LoRa puede emitir el formato
binario de 26 B a un gateway LoRaWAN. El network server (p. ej. ChirpStack)
entrega el payload por MQTT/HTTP a `LocationService.ingest_binary(raw)` con la
hora de recepción del gateway como `received_at`. Se modela cada gateway como un
nodo más (`N0xxx`) dentro de una zona de cobertura amplia en `[zones.*]`.
Consideraciones: la señal LoRa es RSSI/SNR del gateway (resolución de zona
gruesa), el ciclo de trabajo limita la tasa de anuncios (ajustar
`evidence_max_age_s` y `node_timeout_s` para esa zona) y no es necesario para el
primer prototipo: todo funciona solo con BLE.

## Simulador para pruebas

`location.simulator.ZoneNodeSimulator` genera observaciones deterministas
(modelo log-distancia con semilla) con inyección de caídas de nodo
(`set_outage`) y relojes desviados (`set_clock_offset`). Valida lógica, no el
canal de radio real.

## Pendiente de hardware

- Firmware ESP32-S3 que emita este contrato y lea anuncios BLE.
- Calibración de `tx_power`, exponente de pérdida y umbrales con mediciones reales.
- Agente AgentScope que consuma `LocationService` y publique en `Topic.LOCATION`
  (no se registró en `MULTIAGENT_ROUTE` para no alterar la topología acordada).
- Repositorio de personal persistente y gestión de la clave de seudonimización.
