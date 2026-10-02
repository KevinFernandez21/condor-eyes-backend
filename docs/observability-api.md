# API de observabilidad (rol `comms`)

API **de solo lectura** que muestra qué hace el sistema multiagente: agentes, tráfico del bus, eventos y decisiones. Vive en `src/comms/` y la arranca/detiene el `AgentRuntime` como manejador del rol `comms` (`AgentKind.TOOL`). Cubre la parte de lectura de #18; confirmaciones (ack), SQLite y retención de clips siguen en #18.

## Reglas

- Solo metadata JSON finita ya validada por el bus (`envelope_to_dict` revalida: sin NaN/Inf, binarios ni arrays). Nunca frames, imágenes ni embeddings.
- Los identificadores de tags y personas salen **tal como viajan por el bus (seudonimizados)**. Esta capa no tiene tabla de resolución ni forma de revertirlos.
- El tap se suscribe a todos los tópicos mediante `MetadataHub` (nunca AgentScope directo) y es un consumidor síncrono y barato.
- **Vista previa de video: fuera de alcance.** Un endpoint MJPEG/JPEG se sirve del lado del pipeline (limitado en tasa, desactivado por defecto), porque ningún frame cruza el bus. Esta API no tiene ruta de video (hay una prueba que lo verifica).
- `uvicorn` se importa de forma diferida al arrancar; importar `comms` no lo carga.

## Seguridad

- Bind por defecto: `127.0.0.1`. Un host no loopback **sin token** lanza `ConfigurationError` al construir el manejador/app (la API se niega a arrancar).
- Token (comparación en tiempo constante sobre bytes UTF-8; un token con caracteres no ASCII nunca provoca error 500): cabecera `Authorization: Bearer <token>` o `X-API-Token: <token>`. Sin token válido: 401 en HTTP, cierre `1008` en el WS. En HTTP no se acepta por query.
- Token en el WebSocket, en orden de preferencia:
  1. Cabecera (`Authorization` / `X-API-Token`) para clientes que no son navegador.
  2. **Subprotocolo** `Sec-WebSocket-Protocol: token.<valor>` (navegadores: `new WebSocket(url, ["token.<valor>"])`; el servidor lo devuelve en el handshake). No aparece en la URL.
  3. `?token=<valor>`: **puede quedar en logs de acceso, historial y proxies**; solo para pruebas locales.
- **Origin del WebSocket**: si la petición trae `Origin` y no está permitido, se cierra con `1008` antes de aceptar (evita el secuestro entre sitios, CSWSH, en el bind local sin token). Sin cabecera `Origin` (clientes que no son navegador) se permite. Por defecto se permiten `http(s)://localhost`, `127.0.0.1`, `[::1]` y el host de servicio, en el puerto en que se sirve la API. Se configura con `allowed_origins=[...]` (reemplaza la lista por defecto; p. ej. el origen del dashboard).
- Con token configurado, `/docs`, `/redoc` y `/openapi.json` se desactivan (404).

## Adaptador `SystemView`

El system runner (#41) puede implementarlo; por defecto `TapSystemView(tap, runtime)`.

```python
class SystemView(Protocol):
    def agents(self) -> list[dict]: ...      # ver /agents
    def topics(self) -> list[dict]: ...      # ver /topics
    def events(self, limit=50, *, stream_id=None, zone=None) -> list[dict]: ...
    def decisions(self, limit=50, *, stream_id=None, zone=None) -> list[dict]: ...
    def listen(self, callback: Callable[[Topic, dict], None]) -> Callable[[], None]: ...
        # callback síncrono y NO bloqueante; devuelve la función para quitarlo

class RuntimeProbe(Protocol):                 # lo que TapSystemView pide al runtime
    def health(self) -> Mapping[str, Mapping]: ...
    def queue_depths(self) -> Mapping[str, int]: ...   # AgentRuntime.queue_depths()
```

`create_app(view, *, token=None, host="127.0.0.1", ws_queue_size=256, allowed_origins=None)` devuelve la app FastAPI. `ObservabilityCommsHandler(hub, host, port, token, sink=None)` crea tap + vista + app + servidor; llamar `bind_runtime(runtime)` tras crear el `AgentRuntime`.

## HTTP

Todos `GET`, JSON. `limit` 1..500 (defecto 50); más nuevo primero.

| Ruta | Respuesta |
|------|-----------|
| `/health` | `{status: "ok"\|"degraded", agents_total, agents_running, agents_failed, components_total, components_ok, components_degraded, uptime_s, ws_clients, ws_dropped_total}` (`degraded` si algún agente está `failed`). `agents_*` cuentan solo los roles; las entradas con `role: "component"` (componentes del ejecutor) van aparte: `components_ok` = `ok`/`simulated`, `components_degraded` = `degraded`/`offline`/`failed` |
| `/agents` | `{agents: [{name, role, instance, state, processed, duplicates, failures, retries, last_error, last_heartbeat, restarts, queue_depth}]}`. `last_heartbeat` = `created_at` del último `system.health` de ese agente (o `null`); `restarts` = comandos `restart` vistos para el rol |
| `/topics` | `{topics: [{topic, count, rate_per_s, window_s, drops, last_message_at}]}` para los 7 tópicos. `rate_per_s` en ventana deslizante (10 s). `drops` = reportes de `queue_overflow` (`drop_oldest`) del hub: cota inferior, porque el hub limita sus reportes |
| `/events?limit&stream_id&zone` | `{events: [Envelope]}`: todo lo publicado en `events` |
| `/decisions?limit&stream_id&zone` | `{decisions: [Envelope]}`: subconjunto de `events` con `payload.decision_id` (decisiones de fusión) |

`zone` filtra por `payload.zone_id` (o `payload.zone`); `stream_id` por el del envelope (o `payload.stream_id`).

`Envelope` = salida de `envelope_to_dict`: `{topic, source, payload, stream_id, created_at, schema_version, payload_version, event_id, correlation_id, causation_id}`.

## Trazas causales (`/traces`, issue #44)

Responde "por qué se levantó esta alerta": reconstruye la cadena `detecciones → tracks → (identidad / ubicación) → decisión de fusión → evento` con latencia por salto. Vive en `src/tracing/` (`TraceStore`); `ObservabilityCommsHandler` lo registra como oyente del tap, `create_app(view, traces=store)` expone las rutas (con el mismo token que el resto de rutas HTTP).

| Ruta | Respuesta |
|------|-----------|
| `/traces?limit` | `{traces: [Resumen]}`, la correlación más reciente primero |
| `/traces/{correlation_id}` | `Traza` completa; `404` si no existe o ya fue expulsada |

```jsonc
// Resumen
{"correlation_id", "started_at", "ended_at", "hops": 3, "topics": ["events", ...],
 "has_alert": true, "end_to_end_ms": 80.6, "decision_ids": ["dec-0025"],
 "truncated": false, "clock_anomaly": false}

// Traza = Resumen sin "hops" numérico, más:
{"hops": [ {"event_id", "correlation_id", "topic", "source", "stream_id",
            "causation_id", "parent_event_ids": ["..."], "created_at",
            "hop_latency_ms": 20.35, "since_start_ms": 20.35, "via": "own"|"upstream"} ],
 "decisions": [ {"event_id", "decision_id", "outcome", "confidence", "reason_codes",
                 "evidence": [ {"evidence_id", "kind", "role", "resolved": true} ]} ]}
```

Semántica:

- `hops` va en orden causal: por `created_at` y, a igualdad de marca, el padre siempre antes que el hijo. Cada salto trae `topic`, `source`, `event_id`, `causation_id` y `created_at`; **nunca el payload**.
- Padres de un salto (`parent_event_ids`): su `causation_id` y, en una decisión de fusión, cada `evidence_id` que el almacén reconoce. `hop_latency_ms` = `created_at` del salto menos el del padre **más reciente**; `null` si es raíz o el padre no está en el almacén. `since_start_ms` = desde el primer salto de la cadena.
- La traza de una decisión incorpora hacia arriba (`via: "upstream"`) la cadena de cada evidencia resuelta, aunque viva en otra correlación. `resolved: false` = el `evidence_id` no está (nunca llegó o fue expulsado).
- `has_alert` = hay algún mensaje en `events` en la cadena. `end_to_end_ms` = último `events` menos el primer salto; `null` sin alerta.
- La latencia sale de `created_at` de los envelopes y supone **un reloj común** entre productores. Si algún salto da negativo, `clock_anomaly: true`. La resolución es la del reloj de pared (en Windows ~15 ms; en Linux/Jetson, microsegundos).
- Acotado: últimas `max_traces` correlaciones (200, LRU por último mensaje) y `max_hops` saltos por correlación (500, `truncated: true` si se pasa). `system.health` no se indexa: cada latido abriría su propia correlación y expulsaría las cadenas que importan.
- Solo metadata; se pierde al reiniciar (en memoria).

## Langfuse (opcional)

Exporta una traza por decisión a Langfuse **solo si el entorno lo configura**. Sin configuración el sistema corre igual (no se crea ningún cliente).

```bash
uv sync --extra observability          # instala el SDK (langfuse>=4)
export LANGFUSE_PUBLIC_KEY=pk-lf-...   # solo por entorno, nunca en código ni en el repo
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=http://localhost:3000   # obligatorio (o LANGFUSE_BASE_URL)
```

- **El host es obligatorio a propósito.** El SDK usa Langfuse Cloud por defecto; si faltara, la metadata saldría del sitio sin que nadie lo decidiera. Si falta, queda desactivado.
- **Autohospedado (recomendado).** Despliegue de Langfuse con Docker Compose: guía oficial en <https://langfuse.com/self-hosting/docker-compose> (no se vendoriza aquí). Mantiene la metadata dentro de la red del sitio.
- **Langfuse Cloud:** si se apunta a la nube, la metadata filtrada (ver abajo) **sale del sitio** hacia un tercero. Decisión de despliegue, no del código.
- Verificado contra `langfuse` 4.16.0 (`Langfuse(public_key, secret_key, base_url)`, `create_trace_id(seed=)`, `start_observation(trace_context=, name=, as_type="span", input=, output=, metadata=)`, `.start_observation()` hijo, `.end()`, `flush()`, `shutdown()`). `LANGFUSE_HOST` está marcado como obsoleto en ese SDK en favor de `LANGFUSE_BASE_URL`; se aceptan ambos.

### Qué se traza (y qué no: estado real de los agentes)

`event` y `supervisor` son hoy **manejadores de reglas** (`agents.handlers`); su ruta declara `AgentKind.REACT`, pero **no hay ningún `ReActAgent` con LLM instanciado**. No se inventan generaciones ni tool calls. Se trazan como spans de decisión por reglas:

| Traza | Cuándo | Contenido |
|-------|--------|-----------|
| `fusion.decision` | mensaje en `events` con `decision_id` | `input.chain` = saltos causales; `output` = resultado filtrado + `end_to_end_ms`; un span hijo `evidence.<kind>` por referencia |
| `event.rule` | otro mensaje en `events` con `source="event"` | ídem, `metadata.decision_engine="rules"`, `llm=false` |
| `supervisor.command` | comando en `system.commands` con `source="supervisor"` | ídem |

`trace_id` determinista (`create_trace_id(seed=event_id)`): reentregar el mismo mensaje no duplica. Cuando existan agentes ReAct, sus llamadas al modelo se enchufarán como observaciones `generation` hijas del span de decisión (pendiente).

### Filtro de privacidad (`src/tracing/privacy.py`)

Allowlist aplicada **antes** de cualquier envío; lo no nombrado no sale:

- Campos permitidos: `decision_id`, `evaluated_at`, `outcome`, `confidence`, `reason_codes`, `track_ref`, `stream_id`, `zone_id`, `requires_operator`, `type`, `action`, `target`, `state`, `role`, `stage`, `error_type`, `count`, `kind`, `status`, y de `evidence` solo `evidence_id`, `kind`, `role`, `observed_at`, `confidence`, `stream_id`, `zone_id`. Del envelope: `topic`, `source`, `stream_id`, `created_at`, `event_id`, `correlation_id`, `causation_id`.
- `person_id` solo sale si tiene forma de seudónimo (`p-<hex>`, `anon-<hex>`, `ps-<hex>`, 4-64 hex); cualquier otro valor se reemplaza por `[redacted]`. Los IDs de tag crudos no están en la allowlist y se descartan.
- Listas de objetos (detecciones, tracks) se reducen a `<campo>_count`. Embeddings, imágenes, recortes de rostro, nombres, cajas y el texto libre `detail` se descartan.
- Los valores permitidos se revalidan: sin estructuras en campos escalares, sin `data:` URIs, cadenas truncadas a 120.
- Cubierto por `tests/test_tracing_privacy.py` y `tests/test_tracing_langfuse.py` (payload hostil con embeddings, imagen base64, tag y nombre: nada llega al cliente, ni con un SDK real y exportador en memoria).

Limitación: el sistema aún no genera seudónimos; `person_id` se asume ya seudonimizado por contrato (docs/evidence-fusion.md). El filtro reconoce la *forma*, no puede probar que el valor sea un hash real.

## WebSocket `/ws`

Query: `topics=events,system.health` (coma; defecto: todos; tópico inválido: cierre `1008`), `stream_id=cam-1`, `token=...` (ver Seguridad; preferir cabecera o subprotocolo).

Mensajes servidor → cliente (lo que envíe el cliente se ignora):

```json
{"type": "hello", "topics": ["events"], "stream_id": null, "queue_size": 256}
{"type": "envelope", "data": { /* Envelope */ }}
{"type": "lag", "dropped": 12, "dropped_total": 40}
```

`lag` avisa de mensajes descartados para ese cliente antes del siguiente `envelope`.

### Cliente lento o muerto

Cada cliente tiene una cola acotada (`ws_queue_size`, 256) con **descarte del más antiguo** y contador. El tap solo hace `push` O(1) síncrono; únicamente la tarea de envío de ese cliente espera al socket. Un socket que falla cierra su canal y quita su oyente. Cubierto por `tests/test_comms_ws_backpressure.py` (5 000 publicaciones con un cliente bloqueado sin frenar el bus).

## Prueba manual

```bash
uv run python scripts/run_comms_demo.py            # http://127.0.0.1:8000/docs
```

Arranca un `AgentRuntime` con `InMemoryHub`, un publicador sintético (2 cámaras, decisiones con IDs seudonimizados) y la API.

## Pendiente

- Integrar con `SystemApp` del runner (#41) implementando `SystemView` allí o pasando `runtime`.
- Dashboard (#43) consume estos esquemas (incluida la vista de trazas).
- Langfuse no ejecutado contra un servidor real (sin claves en la máquina de desarrollo); probado con el SDK y un exportador en memoria.
- Generaciones LLM de los agentes ReAct `event`/`supervisor`, cuando existan.
- Endpoint de vista previa en el pipeline (aparte).
