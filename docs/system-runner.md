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
| `sim` | `fake` 640x480 | `moving` (2 personas sintéticas) | `sim` | `sim` | `sim` |
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
