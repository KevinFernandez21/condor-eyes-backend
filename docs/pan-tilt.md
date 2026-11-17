# Nodo Pan-Tilt ESP32-S3 (issue #17)

Control seguro de dos servos MG995 (pan y tilt) para que una cámara crítica siga
un track visual seleccionado. Sin roll, sin entrenamiento de modelos y sin
respuesta autónoma a amenazas.

> **Estado del hardware: sin validar en banco.** No hay ESP32-S3 ni servos MG995
> disponibles en esta fase. Todo lo implementado se probó contra un simulador
> determinista. Las métricas del simulador están **etiquetadas como simuladas**
> (`"simulated": true` en la metadata) y no sustituyen las mediciones de banco
> de la sección 8, que están **PENDIENTES**.

## 1. Arquitectura

```
 Plano agentes (host / Jetson)                          Nodo (ESP32-S3)
┌──────────────────────────────────────────┐          ┌───────────────────┐
│ TargetObservation (error en imagen)      │          │ Firmware          │
│   -> TrackingController                  │          │  - valida CRC     │
│        zona muerta, suavizado, límite    │  tramas  │  - secuencias     │
│        de velocidad, límites mecánicos,  │ <------> │  - ESTOP enclavado│
│        timeout, vuelta a neutro          │  serie/  │  - watchdog       │
│   -> PanTiltActuator (adaptador)         │  UDP     │  - PWM 50 Hz      │
│        seq, acks, salud, métricas        │          └─────────┬─────────┘
│   -> Transport (serie | simulado)        │                    │ PWM
└──────────────────────────────────────────┘             2 x MG995 (alim. aparte)
```

Módulos en `src/actuation/`:

| Módulo | Responsabilidad |
|---|---|
| `protocol.py` | Tramas de comando y ack, CRC-16, aritmética de secuencias |
| `config.py` | Carga y validación de `configs/pan_tilt.toml` |
| `controller.py` | Error de imagen -> objetivo angular acotado |
| `adapter.py` | `PanTiltActuator`: secuencias, acks, salud, ESTOP, metadata |
| `transport.py` | Interfaz `Transport` y `SerialTransport` (pyserial diferido) |
| `simulator.py` | Nodo simulado determinista (latencia, tiempo muerto, ruido) |
| `scenarios.py` | Lazo cerrado simulado para pruebas y notebooks |

Reglas de arquitectura: por el bus solo viaja metadata serializable en JSON
estricto (sin `NaN`/`inf`; hay una prueba que lo verifica), nunca frames.
La telemetría del adaptador **no** usa `Topic.COMMANDS`, que queda reservado a
las órdenes entrantes:

| Topic | `kind` | Sentido |
|---|---|---|
| `Topic.EVENTS` | `ptz.command` | Salida: comando emitido al nodo |
| `Topic.EVENTS` | `ptz.alarm` | Salida: alarma crítica (`estop_unconfirmed`, `estop_send_failed`) |
| `Topic.HEALTH` | `ptz.health` | Salida: salud y métricas |
| `Topic.COMMANDS` | `ptz.estop` | Entrada: parada de emergencia, de cualquier fuente |
| `Topic.COMMANDS` | `ptz.release_estop` | Entrada: solo de fuentes autorizadas (`release_sources`, por defecto `supervisor`) |

`source` es una etiqueta dentro del proceso, no una autenticación: la defensa
real es que solo el supervisor publique `ptz.release_estop`. No se añadieron
topics. La cola de metadata sin drenar está acotada (1000 mensajes; se descarta
lo más viejo y se cuenta en `envelopes_dropped`).

## 2. Protocolo de hardware

Es el contrato que debe implementar el firmware del ESP32-S3.

### 2.1 Enlace físico

- **Serie por USB CDC** (puerto nativo del ESP32-S3), 115200 baudios 8N1 como
  valor por defecto. Una trama por línea, terminada en `\n`.
- Alternativa: los mismos bytes por UDP (un datagrama por trama). El
  `Transport` es intercambiable; el protocolo no cambia.

### 2.2 Formato de trama

```
<JSON compacto, claves ordenadas, ASCII>*<CRC16 en 4 hex mayúsculas>
```

CRC-16/CCITT-FALSE (polinomio `0x1021`, init `0xFFFF`, sin reflejo ni xorout)
sobre los bytes exactos del JSON, es decir, todo lo anterior al **último** `*`.
Vector de prueba: `CRC("123456789") = 0x29B1`. Una trama con CRC inválido, JSON
inválido, campo faltante, no numérico o no finito (`NaN`, `Infinity`) se
**descarta sin ack**. Ángulos con 2 decimales; `seq` es uint16. Una trama de más de 256 bytes se
descarta (el receptor serie también descarta líneas más largas y la cola de una
línea que desbordó el búfer, sin interpretarla como trama nueva).

### 2.3 Comandos (host -> nodo)

| `cmd` | Campos | Efecto |
|---|---|---|
| `move` | `seq`, `pan`, `tilt` (grados respecto al neutro), `speed` (grados/s, > 0) | Ir al objetivo sin superar `speed` |
| `estop` | `seq` | Congela en sitio y enclava el estado `estop` |
| `clear` | `seq` | Libera el enclavamiento `estop`; no mueve |
| `hb` | `seq` | Latido: reinicia el watchdog y pide telemetría |

Ejemplo (sin CRC): `{"cmd":"move","pan":12.5,"seq":42,"speed":40.0,"tilt":-3.25}`.

### 2.4 Ack (nodo -> host)

Se responde un ack a cada comando con CRC válido:

```
{"ack":<seq>,"last":<último seq aceptado>,"ms":<ms desde arranque>,"pan":<grados medidos>,"st":"<status>","state":"<estado>","tilt":<grados medidos>}
```

`ms` es uint32 (da la vuelta a los ~49,7 días; el host lo compara en aritmética
modular). `last` es el último `seq` aceptado por el nodo; es opcional salvo en los
acks `stale`, donde es **obligatorio** (lo usa el host para re-sincronizarse).

`st`: `ok` (aplicado), `dup` (mismo `seq` que el último aceptado, sin mover),
`stale` (`seq` anterior, ignorado sin mover), `estop` (rechazado por ESTOP
activo), `rejected` (campos inválidos). `state`: `idle`, `moving`, `failsafe`,
`estop`. `pan`/`tilt` son la posición reportada (con su ruido de medición).

### 2.5 Reglas que debe cumplir el firmware

1. **Secuencias**: un comando se acepta solo si su `seq` es estrictamente
   posterior al último aceptado (RFC 1982, módulo 2^16). Igual -> `dup`;
   anterior -> `stale`; en ambos casos no se mueve nada. El primer comando tras
   el arranque se acepta siempre.
2. **ESTOP**: `estop` se atiende **siempre**, aunque su `seq` sea viejo. Estando
   enclavado, todo `move` se rechaza con `st=estop`. Solo un `clear` con `seq`
   más nuevo lo libera (un `clear` viejo no hace nada).
3. **Límites**: el firmware vuelve a recortar `pan`/`tilt` a los límites
   mecánicos y `speed` a `max_speed_dps`, aunque el host ya lo haya hecho
   (defensa en profundidad). Velocidad máxima por eje.
4. **Watchdog de comunicación**: si pasan `node_watchdog_s` sin ningún comando
   válido, el nodo **congela** en la posición actual (`state=failsafe`). No
   vuelve al neutro por sí solo: moverse sin supervisión es peor que quedarse
   quieto. Un comando válido posterior recupera el control (salvo ESTOP).
5. **Reinicio del nodo** (brownout): al volver, `ms` reinicia en 0 y el último
   `seq` se olvida. El host detecta un retroceso grande de `ms` (mayor que
   `max(500 ms, comms_timeout_s)`), acepta la nueva posición, re-sincroniza el
   controlador y cuenta `node_reboots`; retrocesos pequeños son acks reordenados
   y se ignoran. El enclavamiento de ESTOP **del host** se conserva aunque el
   nodo haya olvidado el suyo.
6. **Reinicio del host**: el nodo conserva su último `seq` y rechazaría como
   `stale` todo lo del host reiniciado (que empieza en 0). Mientras el nodo no
   haya aceptado ningún comando de esta sesión, un ack `stale` con `last` sobre un
   comando propio hace que el host adopte `seq = last` y reenvíe el neutro de
   arranque (y un `clear` pendiente) con `seq = last + 1`. No debilita la
   protección contra repeticiones: nunca retrocede, solo ocurre antes de
   sincronizar y el siguiente `seq` es siempre posterior al último aceptado. Un
   `stale` posterior ya sincronizado nunca cambia la secuencia (cuenta
   `seq_resyncs`).
7. **Arranque seguro**: al energizar, adjuntar los servos solo después de
   escribir el pulso neutro; objetivo inicial = neutro; sin movimiento hasta el
   primer comando válido.

## 3. Comportamiento de seguridad (host)

| Situación | Comportamiento |
|---|---|
| Error dentro de la zona muerta | No se mueve |
| Error grande | Paso limitado por `velocidad * dt`; objetivo recortado a límites |
| Observación vieja (> `target_timeout_s`), ausente o no finita | Se trata como objetivo perdido |
| Objetivo perdido | Congela en la posición medida; tras `hold_s` vuelve al neutro a `neutral_speed_dps` |
| Sin acks por `comms_timeout_s` | Salud `lost`: el host deja de emitir movimientos (sigue enviando latidos) |
| Ack tardío o fuera de orden | No rebobina la posición medida (la telemetría solo avanza por `ms`) |
| Enlace recuperado / ESTOP liberado | El controlador se re-sincroniza con la posición real antes de seguir |
| ESTOP | El host se enclava **antes** de enviar: aunque el transporte falle (la excepción se propaga), no se emiten movimientos. Reintenta con el mismo `seq` hasta confirmación; no hay movimiento hasta `release_estop()` |
| ESTOP sin confirmar tras `estop_retries` | Alarma explícita: evento `ptz.alarm` (`severity: critical`) en `Topic.EVENTS`, log de error, `alarms` en las métricas y `estop_unconfirmed`; el host sigue enclavado |
| Fallo del transporte (`TransportError`/`OSError`) | Se **propaga** desde `poll()`/`track()`: el llamador debe tratarlo. No se envía nada y el watchdog del nodo (`node_watchdog_s`) lo congela en sitio |

Todo comando emitido pasa por un recorte final en el adaptador, independiente
del controlador. Las pruebas de propiedades (cientos de entradas aleatorias,
incluyendo basura) verifican que ninguna salida excede límites ni velocidad.

## 4. Configuración

`configs/pan_tilt.toml`, secciones `[pan_tilt.limits]`, `[pan_tilt.control]`,
`[pan_tilt.comms]` y `[pan_tilt.simulator]`. Claves desconocidas o valores
incoherentes (neutro fuera de límites, velocidad de seguimiento mayor que el
techo duro, watchdog menor que el latido, etc.) lanzan `ValueError`. Los valores
son un punto de partida **conservador y sin validar en banco**; el `kp = 0.4`
se eligió en simulación (con `kp = 0.6` el sobrepaso simulado es de ~17 %).

## 5. Alimentación y cableado

- **Alimentación separada de los servos.** Los MG995 deben ir en un riel propio
  (5-6 V, según su hoja de datos), dimensionado para la corriente de bloqueo de
  **ambos** servos a la vez con margen. No alimentarlos desde el pin 5V/3V3 del
  ESP32-S3 ni desde el USB del host: un servo bloqueado causa caídas de tensión
  y reinicios del microcontrolador. La corriente real se mide en banco (sección 8).
- **Masa común.** Unir la masa del riel de servos con la del ESP32-S3; sin
  referencia común la señal PWM no es interpretable.
- Condensador de reserva (p. ej. 470-1000 uF) cerca de los servos y cables de
  potencia cortos y gruesos, separados de la señal.
- Señal de los servos desde GPIO sin función de arranque (evitar pines de
  strapping) y, si el servo no tolera 3,3 V como nivel alto, usar adaptador de
  nivel (verificar en banco).
- El ESP32-S3 se alimenta por USB desde el host o por un regulador independiente
  del riel de servos; el USB solo transporta datos y la lógica del nodo.
- Interruptor o conector de corte del riel de servos accesible: es el último
  recurso de seguridad física, por encima del software.

## 6. Uso

```python
import sys; sys.path.insert(0, "src")
from actuation.adapter import PanTiltActuator
from actuation.config import load_actuation_config
from actuation.controller import TargetObservation
from actuation.transport import SerialTransport

cfg = load_actuation_config("configs/pan_tilt.toml")
act = PanTiltActuator(SerialTransport("COM5"), cfg, camera_id="cam-01")
act.start()                                   # ordena el neutro a baja velocidad
act.track(TargetObservation(0.2, -0.1, t))    # un ciclo de control (~20 Hz)
envelopes = act.drain_envelopes()             # (Topic, MetadataEnvelope) a publicar
```

`SerialTransport` requiere `pyserial`, que **no** es dependencia del proyecto
todavía; se importa de forma diferida y falla con un mensaje claro si falta
(`uv add pyserial` al integrar el hardware).

Simulación (sin hardware):

```python
from actuation.scenarios import ClosedLoopSim
sim = ClosedLoopSim(cfg, lambda t: (30.0, 10.0), vision_latency_s=0.1)
trace = sim.run(10.0)          # resultados SIMULADOS
print(sim.actuator.metrics())  # incluye "simulated": true
```

## 7. Métricas registradas

`PanTiltActuator.metrics()` y la metadata `ptz.health` incluyen: latencia de
comando->ack (último, media, p95, máximo, muestras), error de seguimiento
normalizado (último y media), ruido de posición observado (desviación estándar
de la posición reportada con el nodo en reposo), contadores (movimientos,
latidos, acks por estado, timeouts, tramas corruptas, movimientos suprimidos) y
salud del enlace (`unknown`, `healthy`, `degraded`, `lost`, `estop`). El campo
`simulated` distingue simulación de hardware.

## 8. Procedimiento de banco (PENDIENTE)

Requiere ESP32-S3 con el firmware del protocolo, 2 x MG995 montados con la
cámara de carga real, riel de servos separado y un osciloscopio o analizador
lógico, más un multímetro/pinza amperimétrica. **Ningún valor de esta tabla se
ha medido todavía.** No rellenar con estimaciones.

Precondiciones: límites del TOML reducidos a la mitad del recorrido físico
libre en la primera prueba; interruptor de corte del riel al alcance.

1. **Arranque seguro**: energizar con el host apagado y verificar que los
   servos van al neutro sin sacudidas y que no se mueven hasta el primer comando.
2. **Latencia**: enviar 1000 `hb` a 20 Hz y registrar latencia comando->ack
   (media, p95, máximo) y pérdida de tramas.
3. **Velocidad real**: escalones de 10/30/60 grados por eje con `speed` de 15,
   30 y 60 dps; medir velocidad y tiempo muerto con vídeo a alta cadencia o
   encoder externo. Confirmar que nunca supera `max_speed_dps`.
4. **Estabilidad**: seguir un objetivo fijo 10 min con el lazo real; contar
   inversiones de sentido y amplitud pico a pico en reposo.
5. **Ruido**: con el nodo en reposo, registrar la desviación estándar de la
   posición reportada y el jitter visible en la imagen.
6. **Alimentación**: medir caída de tensión del riel y corriente pico durante
   escalones de ambos ejes simultáneos y con el servo bloqueado un instante;
   verificar que el ESP32-S3 no se reinicia.
7. **Seguridad**: ESTOP (tiempo hasta detención), corte de enlace (el nodo debe
   congelar en `node_watchdog_s`), comandos duplicados/viejos reinyectados.
8. **Límites mecánicos**: ordenar ángulos fuera de rango y confirmar que el
   firmware recorta y no hay golpe contra el tope.

| Medición | Simulado (no es hardware) | Banco |
|---|---|---|
| Latencia comando->ack | configurable (`one_way_latency_s`) | PENDIENTE |
| Velocidad real de movimiento | supuesto `servo_speed_dps` | PENDIENTE |
| Estabilidad / oscilación | pruebas de `tests/test_actuation_closed_loop.py` | PENDIENTE |
| Ruido de posición | supuesto `noise_std_deg` | PENDIENTE |
| Consumo / caída de tensión | no modelado | PENDIENTE |

## 9. Criterios de aceptación (issue #17)

| Criterio | Estado |
|---|---|
| Comandos nunca exceden límites mecánicos ni de velocidad | Cumplido (pruebas de propiedades; recorte en controlador, adaptador y nodo simulado) |
| Objetivo perdido y comunicación perdida detienen el movimiento | Cumplido en simulación |
| Duplicados y fuera de orden no producen movimiento descontrolado | Cumplido en simulación (secuencias, replay, jitter) |
| Pruebas de simulador: convergencia, oscilación, timeout, ESTOP | Cumplido |
| Resultados de banco (velocidad, estabilidad, ruido, alimentación) | **PENDIENTE** (sección 8) |
| PR enlaza el issue e incluye el protocolo de hardware | Este documento (sección 2) |
