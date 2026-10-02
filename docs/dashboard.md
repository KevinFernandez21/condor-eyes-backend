# Dashboard del sistema (#43)

Una pantalla para ver el sistema multiagente funcionando: grafo de agentes, mensajes en vivo, mapa de sitio, alertas de fusión y salud. **Solo es cliente de la API de observabilidad** (`docs/observability-api.md`): nunca toca el bus ni el runtime.

## Ejecutar

```bash
uv run python scripts/run_dashboard.py --demo      # demo completa; abre http://127.0.0.1:8080
```

`--demo` arranca `scripts/run_comms_demo.py` (runtime con `InMemoryHub`, tráfico sintético y API en `127.0.0.1:8000`) y el dashboard en `127.0.0.1:8080`. Todo escucha solo en loopback.

Contra una API ya en marcha:

```bash
CONDOR_API_URL=http://127.0.0.1:8000 CONDOR_API_TOKEN=... uv run python scripts/run_dashboard.py
```

| Variable / opción | Efecto |
|---|---|
| `CONDOR_API_URL` / `--api-url` | Base de la API (defecto `http://127.0.0.1:8000`) |
| `CONDOR_API_TOKEN` | Token de la API; se envía por cabecera `Authorization: Bearer`, también en el WebSocket. Nunca va en la URL ni llega al navegador |
| `CONDOR_SITE_MAP` / `--site-map` | `site_map.toml` opcional (defecto `configs/site_map.toml`; si no existe, el panel lo indica) |
| `--host`, `--port`, `--allow-lan` | El dashboard no tiene login propio: fuera de loopback exige `--allow-lan` |

## Arquitectura

```
navegador ──GET /api/state (1 s)──▶ servidor del dashboard ──HTTP /health /agents /topics /decisions──▶ API #42
                                       └─────────────── WebSocket /ws (backoff) ───────────────────────▶
```

- El **servidor del dashboard** hace de cliente de la API (HTTP cada 1 s + `/ws` con reconexión por backoff 1→15 s) y mantiene un estado acotado en memoria. El navegador solo consulta `/api/state`, así que **no hay CORS ni problema de `Origin`**: el WS lo abre un cliente sin cabecera `Origin`, que la API permite.
- `src/dashboard/viewmodel.py`: funciones puras (grafo, feed, alertas, mapa, salud, redacción). `state.py`: estado. `client.py`: red. `server.py`: FastAPI. `static/`: HTML/CSS/JS que solo pinta (nunca inserta HTML con datos del bus).
- Si la API cae: banner, estado `Sin conexión`, el grafo se atenúa y se conservan los últimos datos; el WS reintenta solo. Los mensajes `lag` se cuentan y se avisan.

## Paneles

- **Grafo**: nodos = roles de `src/agents/route.py` + `fusion`, `location`, `identity`, `actuation` (previstos, punteados hasta que la API los reporte); aristas = tópicos con msg/s y descartes de `/topics`. Color por estado: en marcha, degradado, caído, detenido, sin datos (también con texto).
- **Mensajes**: filtro por tópico, búsqueda, `payload` expandible, enlace por `correlation_id`.
- **Alertas**: resultado, códigos de razón, evidencia y `requires_operator` (siempre visible).
- **Mapa de sitio**: zonas y receptores de `site_map.toml` (opcional); presencia de tags desde mensajes de localización (`tag_ref`, `status`, `zone_id`); sin ellos: «sin datos».
- **Salud**: `/health` y salud por stream; FPS, latencia p50/p90, CPU y memoria muestran «sin datos» si el pipeline no los publica (hardware: laptop).
- **Cámara**: marcador; la vista previa vive del lado del pipeline y no está implementada (ningún frame pasa por la API).

## Privacidad

Se muestran los identificadores tal como llegan (seudonimizados). Defensa en profundidad: claves tipo `embedding`, `image`, `crop`, `jpeg`, `b64` y vectores numéricos largos se sustituyen por `[omitido]`.

## Pendiente

- Vista previa de cámara con cajas (endpoint del pipeline).
- CPU/memoria/FPS reales cuando el pipeline los publique.
- Tópico `location` en el bus (hoy se reconoce por tópico o por forma del payload).
