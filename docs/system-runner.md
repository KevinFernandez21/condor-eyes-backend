# Ejecutor local del sistema (issue #41)

Un solo comando levanta **todo el sistema multiagente en el portátil**, con
entradas reales o simuladas, para verlo funcionar antes de tener la Jetson.

```bash
uv run python scripts/run_system.py --profile sim              # sin hardware
uv run python scripts/run_system.py --profile laptop           # webcam + tag C6
uv run python scripts/run_system.py --profile replay           # video + JSONL grabados
uv run python scripts/run_system.py --profile sim --duration 10 # termina solo y resume
```

Ctrl+C (o Ctrl+Break en Windows) detiene todo en orden y devuelve código 0. Un
perfil o archivo de configuración inválido devuelve código 2.

## Qué arranca

```
plano de video (hilos)                      plano de agentes (asyncio, un proceso)
fuente ─► LiveVideoPipeline ─► Detector ─►  MetadataPublisher ─► InstrumentedHub (InMemoryHub)
 fake | webcam | archivo        (lado del pipeline)                │
                                                                  ├─► AgentRuntime: ingest, inference,
                                                                  │   tracker, event, storage, comms, supervisor
                                                                  ├─► FusionService ─► decisiones en EVENTS
                                                                  └─► location / identity / actuation
                                                                      (simulados o plugin real)
```

- El **detector** corre en el hilo del pipeline (`processor`); del frame solo
  salen detecciones (`xyxy`, `cls`, `conf`) como números finitos. Ningún frame,
  imagen ni embedding llega al bus.
- El rol `tracker` usa un `SimpleTracker` por IoU y asigna `zone_id` por
  posición horizontal (zonas en `configs/system.toml`). No sustituye a
  NvDCF/ByteTrack (ARCH-003).
- El rol `event` emite `track.zone_entered` al cambiar un track de zona.
- `FusionService` evalúa cada `fusion_interval_s` y publica en `Topic.EVENTS`
  solo las decisiones que cambian.

## Perfiles (`configs/system.toml`)

| Perfil | Cámara | Detector | Tag | Identidad | Actuador |
|---|---|---|---|---|---|
| `sim` | `fake` x4 (1920x1080, `[[cameras]]`) | `scenes` (personas, vehículos, movimiento) | `sim` | `sim` | `sim` |
| `laptop` | `webcam` (índice 0) | `yolov8n` (`configs/surveillance.toml`) | `c6` | `none` | `sim` |
| `replay` | `file` (`data/replay/clip.mp4`) | `yolov8n` | `replay` (`data/replay/tags.jsonl`) | `none` | `none` |

La configuración se valida de forma estricta: clave desconocida, `nan`/`inf`,
booleano donde va un número, `kind` fuera de catálogo, zonas duplicadas o con
rango inválido abortan el arranque (`ConfigError`). Las rutas relativas se
resuelven contra la raíz del repositorio.

Formato del JSONL de `replay` (una estimación por línea; `offset_s` opcional,
se repite en bucle):

```json
{"offset_s": 0.0, "person_ref": "person-sim-01", "zone_id": "vault", "confidence": 0.9}
```

El video de `replay` se reproduce en bucle a `fps`. Como `StreamSource` solo
acepta RTSP/USB, la fuente de archivo usa la URI placeholder `usb:0` y la
ignora (ver `system/sources.py`).

## Maqueta multicámara (perfil `sim`)

`uv run python scripts/run_system.py --profile sim --api [--seed N]` simula 4
cámaras a 1920x1080 definidas en `[[cameras]]` de `configs/system.toml`
(`cam-01` Entrada, `cam-02` Bodega, `cam-03` Perímetro, `cam-04`
Estacionamiento), cada una con sus zonas (franjas horizontales; los `zone_id`
son únicos entre cámaras y alguna es `restricted`).

- **Escenas**: por cámara aparecen personas, vehículos (`car`, `truck`, `bus`) y
  movimiento sin clase (`cls = -1`, `label = "motion"`) con cajas `xyxy`
  realistas (persona alta, vehículo ancho) que cruzan el encuadre; algunas
  personas se detienen unos segundos. Parámetros en `[profiles.sim.scenes]`.
- **Semilla**: cada cámara tiene su propio generador sembrado con
  `--seed` + `camera_id` y avanza por frame, así que la secuencia de cada
  cámara es reproducible. Sin `--seed` se elige una al azar y se imprime
  (`Semilla de la simulación: N`; también en `snapshot()["seed"]`).
- **Eventos aleatorios**: cada persona tiene un perfil: `staff` (conocida, con
  tag), `staff_no_tag` (conocida, sin tag: "persona sin tag") o `stranger`
  (rostro desconocido y sin tag: intrusión si entra en una zona restringida);
  a veces el rostro queda tapado (`no_face`). Además hay **caídas y
  recuperaciones de cámara** (`dropouts`, `dropout_interval_s`,
  `dropout_duration_s`).
- **Seudónimos**: los identificadores de simulación (`person-sim-01`) nunca
  llegan al bus: se publican como `p-xxxx` (persona) y `tag-xxxxxxxx` (tag),
  HMAC-SHA256 con una clave derivada de la semilla. Los permisos de la fusión se
  seudonimizan igual. En `replay`, el JSONL usa esos identificadores de
  simulación y también se seudonimizan.

### Formas de payload (solo metadata, JSON estricto)

**Detecciones** (`Topic.DETECTIONS`, `stream_id` del envelope = `camera_id`).
Payload de frame del pipeline:

```json
{"stream_id": "cam-01", "frame_index": 120, "timestamp": "2026-10-02T12:00:00.100000+00:00",
 "width": 1920, "height": 1080,
 "detections": [
   {"stream_id": "cam-01", "track_id": 7, "cls": 0, "label": "person", "conf": 0.912,
    "xyxy": [812.4, 402.1, 901.0, 655.3], "frame_ts": "2026-10-02T12:00:00.099000+00:00"}]}
```

`cls` sigue `surveillance.classes.NAMES` (0 persona, 2 coche, 4 bus, 5 camión) o
`-1` para movimiento sin clase (`label: "motion"`). `track_id` es el id de la
fuente (estable mientras el objeto vive).

**Tracks** (`Topic.TRACKS`): `{"stream_id", "frame_index", "timestamp", "tracks":
[{"track_ref": "cam-01/7", "track_id": 7, "stream_id", "cls", "label", "conf",
"xyxy", "zone_id"}]}`.

**Estado de cámara** (`Topic.STREAM_STATUS`, uno por cámara cada
`min(1 s, heartbeat_interval_s)`; `stream_id` = `camera_id`; el pipeline también
publica en este tópico su salud cruda de stream **sin** `kind`: filtre por
`kind == "camera.status"`):

```json
{"kind": "camera.status", "camera_id": "cam-02", "stream_id": "cam-02",
 "name": "Bodega", "scene": "Pasillo y almacén",
 "zones": [{"zone_id": "bodega-pasillo", "restricted": false},
           {"zone_id": "bodega-almacen", "restricted": true}],
 "fps": 10.0, "measured_fps": 9.8,
 "state": "ok", "stream_state": "connected",
 "resolution": {"width": 1920, "height": 1080},
 "frames_received": 1423, "last_frame_at": "2026-10-02T12:00:00.100000+00:00",
 "last_error": null, "simulated": true}
```

`state`: `ok` (conectada), `degraded` (reconectando) u `offline` (3 o más
reintentos fallidos, `failed` o `stopped`). En `laptop`/`replay` hay una sola
cámara con `resolution` en `null`.

**Eventos y decisiones** (`Topic.EVENTS`): `track.zone_entered`
(`{"kind", "track_ref", "zone_id", "label", "from_zone", "restricted",
"stream_id"}`) y decisiones de fusión (`decision_id`, `outcome`,
`reason_codes` como `unknown_face`, `person_without_tag`,
`zone_not_permitted`, `person_id` seudonimizado).

**Telemetría de sensores** (`Topic.HEALTH`, distinguida por `kind`):
`location.estimate`, `identity.result`, `ptz.command`, `ptz.health`,
`component.health`.

### `/health` y `/agents`

`/health` cuenta solo los 7 roles: `agents_total` / `agents_running` /
`agents_failed`. Los componentes del ejecutor se informan aparte:
`components_total`, `components_ok` (`ok`/`simulated`) y `components_degraded`
(`degraded`/`offline`/`failed`). `/agents` los lista con `role: "component"`.

## Degradación elegante

Nada de lo que falta tumba el proceso; se ve en `snapshot()["components"]` y en
`Topic.HEALTH` (`kind: "component.health"`, sin clave `state` para no activar
reinicios del supervisor).

| Falta | Efecto |
|---|---|
| Webcam / video | `camera: degraded` (el pipeline reintenta con backoff) |
| Pesos / `ultralytics` (ruta: `CONDOR_WEIGHTS` > `[detector].weights`, por defecto `weights/yolov8n.pt`; nunca se versionan) | `detector: degraded` (el mensaje indica ruta, `CONDOR_WEIGHTS` y el asset oficial; nunca se descarga nada); detecciones vacías, el resto sigue |
| C6 o `tagbridge` | `location: degraded`; se publica `status: "unknown"` (nunca se inventa zona) |
| JSONL de replay | igual que sin C6 |
| Plugin (`location`, `identity`, `actuation`, `tagbridge`) | `plugin: "not_installed"`; se usa el simulador |

## Plugins opcionales

`system/plugins.py` descubre con `importlib.util.find_spec` los módulos
`location` (#32), `tagbridge` (#37), `actuation` (#33) y `faceid.identity`
(#36). **Hoy ningún perfil conecta el módulo real**: aunque estén instalados,
el ejecutor lo reporta (`plugin: "installed"`) y sigue usando el simulador o el
estado `unknown`; los adaptadores reales se añaden tras integrar esos PR.

## Simuladores y formas de payload

Imitan los payloads de los módulos reales (ver `git show origin/<rama>:docs/...`):

| Simulador | Tópico | `kind` | Forma |
|---|---|---|---|
| ubicación | `LOCATION_TOPIC` (`Topic.LOCATION` si existe; si no `Topic.HEALTH`) | `location.estimate` | `location.bus_adapter.to_envelope` (#32) |
| identidad | `Topic.HEALTH` | `identity.result` | resultado de `faceid.identity` (#36), más `track_ref` |
| actuador | `Topic.HEALTH` | `ptz.command` / `ptz.health` | `actuation.adapter` (#33) |

`Topic.EVENTS` queda reservado para incidentes reales y decisiones de fusión
(sin ruido de sensores). Ubicación, identidad y PTZ son telemetría y viajan por
`Topic.HEALTH` distinguidas por `kind`; `LOCATION_TOPIC` pasa a ser
`Topic.LOCATION` cuando #32 esté en `main`. Todos
llevan `"simulated": true`.

## API de observabilidad (#42) y dashboard (#43)

```bash
uv run python scripts/run_system.py --profile sim --api                 # http://127.0.0.1:8000
uv run python scripts/run_system.py --profile sim --api --api-port 8001
uv run python scripts/run_system.py --profile sim --api --api-host 0.0.0.0 --token <secreto>
```

- Se sirve como parte del ciclo de vida del rol `comms` (`RunnerCommsHandler`):
  arranca el `BusTap` (siempre) y, con `--api`, uvicorn (importación diferida).
- Host `127.0.0.1` por defecto. Cualquier otro host **exige token**
  (`--token` o `CONDOR_API_TOKEN`); sin él el comando sale con código 2. El token
  nunca vive en `configs/system.toml`.
- `[api].allowed_origins` en `configs/system.toml` lista los orígenes web que
  pueden abrir el WebSocket (anti-CSWSH). Por defecto el dashboard en
  `http://127.0.0.1:8080` y `http://localhost:8080`; reemplaza la lista por
  defecto de la API. Si el dashboard se sirve desde otro origen, añádalo ahí.
- Rutas (ver `docs/observability-api.md`): `/health`, `/agents`, `/topics`,
  `/events`, `/decisions`, `/ws`. `/agents` incluye, además de los roles, una
  entrada por componente del ejecutor con `role: "component"`, `name:
  "component/<nombre>"` y `state` = estado del componente (`ok`, `simulated`,
  `degraded`, ...; `last_error` lleva el detalle si está degradado).
- Estadísticas por tópico: **solo** el `BusTap` (el hub es un `InMemoryHub`
  sin contadores propios).

### API Python para otros componentes

```python
import sys; sys.path.insert(0, "src")
from system import SystemApp, load_system_config

app = SystemApp(load_system_config(None, "sim"), api_token=None)
await app.start()           # o: await app.run(duration=10, stop_event=ev)
app.runtime                 # AgentRuntime
app.hub                     # InMemoryHub (history, stats)
app.tap                     # BusTap
app.view                    # SystemView (TapSystemView): agents(), topics(), events(), decisions(), listen()
app.api_url                 # "http://127.0.0.1:8000" o None
app.snapshot()              # {profile, running, uptime_s, components, pipeline, api}
await app.stop()            # idempotente
```

`SystemApp(config, *, hub=None, plugins=None, source_factory=None, detector=None,
api_token=None)`. Estados de componente: `ok`, `simulated`, `starting`,
`degraded`, `disabled`, `not_used`, `stopped`.

## Parada limpia

Orden: latido de salud, pipeline de video (une hilos), componentes, vaciado de
colas del runtime, `runtime.stop()`, cierre del hub, cierre del detector. Las
pruebas verifican que no quedan tareas asyncio ni hilos `video-*`.

## Seguimiento: unificar con `DetectorProcessor` (#40)

Todo el cableado del detector vive en `src/system/detection.py`
(`RunnerDetector`: construir, warmup, `processor` del pipeline, cierre).
Cuando #40 llegue a `main` con `pipeline.detection.DetectorProcessor`, se
sustituye `RunnerDetector` por ese adaptador en un solo punto
(`SystemApp._detection`) y se elimina el duplicado. Se conserva la política
actual: una detección malformada se **descarta y se cuenta**
(`components.detector.malformed_dropped`), sin tumbar el frame; se propondrá a
#40.

## Fuera de alcance

UI, Jetson/DeepStream y Docker Compose (#18).
