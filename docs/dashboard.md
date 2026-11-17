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

## Seguridad y TLS

- Cliente HTTP con `verify=False` **solo para `http://`**: es un rodeo del entorno (en esta laptop Windows el almacén de certificados tiene uno inválido y `ssl.create_default_context()` lanza `INVALID_CERTIFICATE`, incluso al construir un cliente `httpx`). En `http://` no hay TLS que verificar, así que no se pierde ninguna garantía.
- Con `https://` se verifica el certificado con el almacén del sistema. Si falla por el mismo problema, usa `truststore` (almacén del SO) o `certifi` (bundle propio) en lugar de desactivar la verificación.
- **`--allow-lan` con API en `http://`: el token viaja en claro** por la red (cabecera `Authorization`), y el dashboard no tiene login propio. Úsalo solo en redes de confianza o pon TLS delante.

## Vistas

Navegación por pestañas (`#agentes`, `#multicamara`); diseño oscuro tipo VMS, un solo acento y colores de estado/clase como variables CSS. A 375 px la rejilla pasa a una columna y la línea de tiempo se desplaza dentro de su contenedor.

### Agentes

- **Salud**: `/health` más salud por stream. «Agentes en marcha» cuenta solo roles (los componentes no suman). FPS, latencia, CPU y memoria muestran «—» si el pipeline no los publica (hardware: laptop).
- **Componentes**: tira aparte (cámara, detector, fusión…) con la entrada `role: "component"` de `/agents`; no son nodos del grafo.
- **Grafo por capas**: pipeline `ingest → inference → tracker → event → storage`, supervisor y comms debajo, plugins (`location`, `identity`, `fusion`, `actuation`) en un carril lateral (punteados mientras la API no los reporte). Cada nodo muestra msg/s (delta de `processed` entre sondeos), cola y reinicios. Las aristas de control (`system.*`) se ocultan salvo «Mostrar control»; los tópicos van en una tabla plegable. Ningún valor ausente se pinta como `null`/`None`: siempre «—».
- **Alertas**, **mensajes en vivo** (filtro por tópico, búsqueda, `correlation_id`) y **mapa de sitio** (opcional) como antes. Cada decisión nueva de gravedad alta o media lanza un aviso apilado abajo a la derecha («Alerta: persona en entrada»).

### Multicámara (maqueta, sin video)

- Barra con filtros **Estado**, **Zonas** (escena de la cámara) y **Alertas** (solo cámaras con alertas recientes), distribución 1×1 / 2×2 / 3×3 con paginación, «Revisión aleatoria» y «En vivo» (en pausa congela las teselas).
- Cada tesela es una escena sintética en `<canvas>` (determinista por cámara: entrada, estacionamiento, perímetro, bodega, pasillo) con cajas, etiqueta, `track_id` y confianza dibujadas desde la metadata de detección, escaladas desde el tamaño de frame del payload (1920×1080 por defecto). Nombre abajo a la izquierda; cámaras caídas o degradadas muestran un velo. Clic en una tesela la amplía; Esc o clic vuelve.
- **Revisión aleatoria**: en el cliente, cada 6 s elige una cámara al azar (semilla por hora), la resalta y registra en «Revisiones» hora, cámara, número de detecciones y resultado (OK o alerta si la cámara no está en línea o tiene una alerta alta reciente).
- **Línea de tiempo**: se construye en el servidor (`src/dashboard/timeline.py`) con cubetas de un minuto por cámara y clase (máximo simultáneo, acotado a 180 min y 64 cámaras). Carriles Persona (amarillo), Vehículo (naranja) y Movimiento (celeste); el tooltip de cada barra da los conteos, el cursor sigue al ratón y un clic lo fija y pausa. No hay grabación: el cursor lee conteos, no video.
- Metadata de cámara tolerante: `name`/`scene` planos o dentro de `camera` en `stream.status`/`system.health`; sin ella, el nombre es el `stream_id` y la escena se deduce de forma estable. Diseñado para N cámaras.

## Privacidad

Se muestran los identificadores tal como llegan (seudonimizados). Defensa en profundidad: claves tipo `embedding`, `image`, `crop`, `jpeg`, `b64` y vectores numéricos largos se sustituyen por `[omitido]`.

## Pendiente

- **Cámaras reales**: las teselas necesitan un endpoint de instantáneas/MJPEG **del lado del pipeline** (acotado en tasa y desactivado por defecto). Ningún frame pasa por el bus ni por la API; mientras tanto la vista es una maqueta con escenas sintéticas.
- CPU/memoria/FPS reales cuando el pipeline los publique.
- Tópico `location` en el bus (hoy se reconoce por tópico o por forma del payload).
