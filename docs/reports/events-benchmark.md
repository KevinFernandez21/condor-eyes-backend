# Informe: benchmark de eventos (issue #12)

> **Hardware y runtime de estas mediciones:** laptop con Intel Core Ultra 9 275HX (24 hilos), 32 GB de RAM, GPU dedicada NVIDIA GeForce RTX 5080 Laptop (16 GB) y GPU integrada Intel Graphics, Windows 11, Python 3.11.16. Runtime: Python puro en CPU (sin GPU). **No** es la Jetson Orin Nano ni la laptop i7 de 12.ª gen. del equipo: los FPS y latencias de aquí no se transfieren. El presupuesto común sin CUDA está en [`docs/edge-budget.md`](../edge-budget.md) (issue #23).

Corrida del 2026-10-01 en la laptop (CPU). Comando:
`uv run python scripts/events.py benchmark --learned`. Configuración: `configs/events.toml`
(merodeo con umbral de 30 s y desplazamiento máximo de 0,12; intrusión a 0,6 s;
aglomeración de 5 personas durante 5 s).

## Datos del held-out (test)

300 escenarios, **15 horas** simuladas y 355.219 observaciones de track, con
perturbaciones más fuertes que en train/val.

- **Comportamientos:**

  | Tipo | Comportamientos |
  |---|---|
  | Normales | 1542 `transit`, 103 `authorized_worker` |
  | Hard negatives | 148 `slow_transit`, 127 `short_wait`, 123 `border_stand`, 508 `group` |
  | Eventos | 147 `loiter`, 160 `intrusion` |

- **Eventos etiquetados:** 208 intrusiones, 202 merodeos y 102 aglomeraciones.

## Resultados: baseline de reglas

| Evento | Precisión | Recall | F1 | Falsas alarmas/h | TTD p50 | TTD p90 | Severidad correcta |
|---|---|---|---|---|---|---|---|
| Intrusión | 0,874 | **1,000** | 0,933 | 2,0 | 0,8 s | 1,2 s | 100 % |
| Merodeo | 0,786 | 0,980 | 0,872 | 3,6 | 4,4 s | 9,8 s | 100 % |
| Aglomeración | 0,895 | **1,000** | 0,944 | 0,8 | 0,05 s | 0,6 s | 100 % |

- **Latencia por observación:** p50 46 µs, p90 101 µs, p99 180 µs (Python puro, CPU de la laptop).
- **Pico de memoria Python:** 11 MB en todo el held-out.

Con este coste el detector de eventos no compite por memoria con el detector
compartido en la Jetson. Esa medición en la Jetson sigue pendiente.

## Modelo aprendido para merodeo (opcional)

- **Modelo:** MLP 5→16→1 sobre las mismas features que la regla: permanencia,
  ventana, desplazamiento neto, recorrido y rectitud.
- **Entrenamiento:** 246.150 muestras de los 300 escenarios de train, con 16,6 % positivas.
- **Calibración:** umbral y suavizado temporal ajustados por F1 **de evento** en 50
  escenarios de val. Resultado: umbral 0,95 y 25 muestras consecutivas (5 s); val F1 = 0,986.

| Merodeo (test) | Precisión | Recall | F1 | Falsas alarmas/h | TTD p50 | TTD p90 |
|---|---|---|---|---|---|---|
| Regla | **0,786** | 0,980 | **0,872** | **3,6** | 4,4 s | 9,8 s |
| MLP | 0,432 | 0,921 | 0,588 | 16,3 | 4,4 s | 11,8 s |

El MLP casi iguala a la regla en val, pero en el held-out **genera 4,5 veces más
falsas alarmas**. De sus 245 falsas alarmas, 206 coinciden con vibración de cámara
y 140 con tránsito lento. Aprendió la distribución de perturbaciones de train y no
generaliza a una cámara con más vibración y ruido.

### Recomendación: no entrenar por ahora

La regla es más precisa y más robusta ante el cambio de condiciones. Además es
explicable, porque la evidencia son los mismos números que la disparan. Se ajusta
por zona sin datos etiquetados y cuesta microsegundos. Un modelo temporal solo se
justificaría con **datos reales etiquetados** del sitio (ver la guía en
`docs/events-benchmark.md`) y si supera a la regla en ese held-out real. En datos
sintéticos solo puede aprender el simulador.

## Análisis de fallos (regla, held-out)

Las causas se cuentan por evento y un evento puede tener varias.

| Evento | Falsas alarmas / omisiones | Causa dominante | Interpretación |
|---|---|---|---|
| Intrusión | 30 FP | 23 con oclusión y 8 con cambio de ID; 18 son la propia persona intrusa y 12 un trabajador autorizado | **Duplicados**: tras una oclusión mayor que `max_gap_s` o un cambio de ID, el track es nuevo y la intrusión se reporta otra vez. Los 12 del trabajador autorizado salen con severidad `review`, así que no llegan como alerta |
| Merodeo | 54 FP | 35 con vibración de cámara, 32 con grupo, 19 con espera corta, 14 con oclusión | **Grupos y esperas** que se quedan cerca del umbral de 30 s; la vibración reduce el desplazamiento neto aparente |
| Merodeo | 4 FN | Vibración y oclusión con cambio de ID | Un cambio de ID **reinicia la permanencia**, así que el merodeo de esa persona no llega al umbral |
| Aglomeración | 12 FP | Grupos con oclusión | El conteo oscila alrededor del umbral cuando alguien del grupo se ocluye |

Modos de fallo conocidos que conviene validar con video real:

- **Oclusión larga o cambio de ID:** duplica intrusiones y reinicia el merodeo. La
  mitigación propuesta es un *cooldown* por zona que una eventos cercanos en tiempo y
  espacio, o un tracker con re-ID (issue #11).
- **Trabajadores estacionarios:** se detectan, pero bajan a `review` solo si llega la
  metadata `authorized`. Sin el agente de identidad generarían alertas.
- **Grupos:** un grupo que espera junto al umbral de merodeo dispara merodeo
  individual. Hay que ajustar `loiter_dwell_s` por zona según el uso real del lugar.
- **Vibración de cámara:** desplaza los puntos de apoyo. Una vibración de hasta 2,5 %
  del frame no generó intrusiones falsas desde el borde (`border_stand`, a 0,012–0,025 normalizado), pero
  sí afecta al merodeo.

## Limitaciones

- Todos los números son **sintéticos** y no sustituyen una evaluación con video real
  del sitio: las conductas siguen un guion y las perturbaciones son modelos simples.
- Una sola geometría de cámara y de zonas.
- Las falsas alarmas por hora dependen de la densidad de gente del escenario, de 3 a
  18 personas por escenario de 3 min.
