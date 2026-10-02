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
- Token: cabecera `Authorization: Bearer <token>` o `X-API-Token: <token>`. En el WebSocket también `?token=<token>` (los navegadores no pueden poner cabeceras). En HTTP no se acepta por query (acaba en logs). Comparación en tiempo constante; sin token válido: 401 / cierre `1008`.
- Sin comprobación de `Origin` en el WS: en loopback sin token cualquier página local podría leer metadata. Usa token si el dashboard está en otro origen.

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

`create_app(view, *, token=None, host="127.0.0.1", ws_queue_size=256)` devuelve la app FastAPI. `ObservabilityCommsHandler(hub, host, port, token, sink=None)` crea tap + vista + app + servidor; llamar `bind_runtime(runtime)` tras crear el `AgentRuntime`.

## HTTP

Todos `GET`, JSON. `limit` 1..500 (defecto 50); más nuevo primero.

| Ruta | Respuesta |
|------|-----------|
| `/health` | `{status: "ok"\|"degraded", agents_total, agents_running, agents_failed, uptime_s, ws_clients, ws_dropped_total}` (`degraded` si algún agente está `failed`) |
| `/agents` | `{agents: [{name, role, instance, state, processed, duplicates, failures, retries, last_error, last_heartbeat, restarts, queue_depth}]}`. `last_heartbeat` = `created_at` del último `system.health` de ese agente (o `null`); `restarts` = comandos `restart` vistos para el rol |
| `/topics` | `{topics: [{topic, count, rate_per_s, window_s, drops, last_message_at}]}` para los 7 tópicos. `rate_per_s` en ventana deslizante (10 s). `drops` = reportes de `queue_overflow` (`drop_oldest`) del hub: cota inferior, porque el hub limita sus reportes |
| `/events?limit&stream_id&zone` | `{events: [Envelope]}`: todo lo publicado en `events` |
| `/decisions?limit&stream_id&zone` | `{decisions: [Envelope]}`: subconjunto de `events` con `payload.decision_id` (decisiones de fusión) |

`zone` filtra por `payload.zone_id` (o `payload.zone`); `stream_id` por el del envelope (o `payload.stream_id`).

`Envelope` = salida de `envelope_to_dict`: `{topic, source, payload, stream_id, created_at, schema_version, payload_version, event_id, correlation_id, causation_id}`.

## WebSocket `/ws`

Query: `topics=events,system.health` (coma; defecto: todos; tópico inválido: cierre `1008`), `stream_id=cam-1`, `token=...`.

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
- Dashboard (#43) consume estos esquemas.
- Endpoint de vista previa en el pipeline (aparte).
