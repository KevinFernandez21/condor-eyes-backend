# Tag BLE con XIAO ESP32-C6 y receptor del PC

Estado: **firmware y receptor probados con hardware real, integrados con
`LocationService`; umbrales sin calibrar**. Issue #26 (depende de #16, localización
por zonas; ver `docs/location.md`).

Con un solo radio no hay triangulación: lo medible es la **presencia dentro de un
área** (RSSI cerca de un receptor fijo). Fuera de alcance: multi-nodo, LoRaWAN, rostros.

## Topología

```
XIAO ESP32-C6 (tag, lo lleva la persona) ──BLE──▶ adaptador Bluetooth del PC (receptor fijo, junto a la cámara)
                                                       │ src/tagbridge: filtra el tag enrolado, arma payload v1
                                                       ▼
                                 LocationService (src/location): valida, suaviza, estima zona y confianza
                                                       ▼
                                 PresenceView: dentro/fuera con umbral + histéresis
```

El receptor del PC hace el papel del nodo de zona del contrato de #16 (`node = N0001`,
zona `lobby` en `configs/location.toml`). Más adelante puede sustituirse por un segundo
ESP32 sin cambiar el contrato. El tag se enrola por la interfaz `PersonnelRepository`
(`InMemoryPersonnelRepository` en el prototipo); los logs solo muestran seudónimos.
Clave de seudonimización: variable `CONDOR_PSEUDONYM_KEY`; sin ella se usa una clave
efímera.

## Anuncio del tag

Datos de fabricante, company ID `0xFFFF` (reservado para pruebas), 14 bytes little-endian:

| Offset | Tamaño | Campo |
|--------|--------|-------|
| 0 | 2 | magic `"CE"` |
| 2 | 1 | versión (`1`) |
| 3 | 6 | ID del tag (por defecto la MAC Bluetooth del chip) |
| 9 | 4 | `seq` uint32, +1 por anuncio |
| 13 | 1 | batería % (`0xFF` = desconocida) |

El receptor filtra por este formato y por el ID enrolado, **no** por la dirección BLE
(que puede cambiar). Cualquier otro dispositivo se descarta sin loguearlo; solo se
cuenta en `receiver.ignored`. Firmware y build: `firmware/xiao_c6_tag/README.md`.

### `seq` monótono entre reinicios

El anti-replay de #16 es por contador circular de 32 bits por (tag, nodo): un `seq`
menor que el último visto es `replay` y uno 2^31 o más adelante también. Un `seq`
aleatorio al arrancar hacía que cerca de la mitad de los reinicios dejaran al tag
rechazado hasta `retention_s`. El firmware reserva bloques en NVS (`Preferences`,
espacio `cetag`, clave `seqblk`):

1. Al arrancar lee el inicio del bloque guardado (0 en el primer arranque), lo usa
   como `seq` inicial y guarda `inicio + SEQ_BLOCK`.
2. Cuando `seq` alcanza el final del bloque reservado, guarda el siguiente bloque.
3. El valor guardado es siempre mayor que cualquier `seq` ya emitido, así que
   tras cualquier reinicio (también por corte de energía) el `seq` sigue creciendo.

`SEQ_BLOCK = 1000`: un reinicio salta como máximo 1000 (muy por debajo de 2^31) y hay
una escritura en NVS cada 1000 anuncios (unos 100 s a 100 ms). Reflashear sin borrar
NVS conserva el contador; borrar la flash lo reinicia a 0 (el servicio lo rechazaría
como `replay` hasta `retention_s`).
Si NVS falla, el firmware sigue anunciando y avisa por serie (`WARN nvs_begin_failed` /
`nvs_write_failed`): el `seq` deja de ser persistente y vuelve el riesgo de `replay`.
El test comprueba que `SEQ_BLOCK` y la clave NVS del modelo coinciden con el `.ino`.
`tests/test_tag_seq_scheme.py` modela este esquema y prueba que `LocationService`
acepta todos los paquetes de cientos de reinicios, y que el esquema anterior se rechazaba.

## Observación emitida (payload v1)

Igual que en `docs/location.md`:

```json
{"v":1,"ch":"ble","tag":"58E6C515220A","node":"N0001","rssi":-74,"ts":1790914468536,"seq":3007}
```

`bat` se omite si el tag no la reporta. `ts` es la hora del PC en ms UTC. El receptor
no filtra repetidos: Windows reentrega anuncios y `LocationService` los cuenta como
`duplicate` en `service.stats`.

## Receptor en vivo

```bash
uv run python scripts/tag_receiver.py --tag 58E6C515220A            # tabla en vivo
uv run python scripts/tag_receiver.py --tag 58E6C515220A --jsonl    # observaciones v1 aceptadas
uv run python scripts/tag_receiver.py --tag ... --duration 30 --enter-dbm -65 --hysteresis-db 6
```

Salida: RSSI crudo, media suavizada por `LocationService` (`smoothing` en
`configs/location.toml`), zona, `inside`/`outside`, confianza `[0, 1]` de la
estimación y `seq`. Los rechazos no se imprimen: salen por motivo en el resumen final.
La vista dentro/fuera entra cuando el RSSI suavizado es `>= enter_dbm` y sale cuando es
`< enter_dbm - hysteresis_db`; sin evidencia fresca pasa a `outside` con el motivo de
`LocationService` (`stale_evidence`, `node_outage`). El ID del tag se pasa por
argumento; no se versiona.

## Site map

`configs/site_map.toml` une cada zona con su receptor (`[[receivers]]`), su cámara
(`[[cameras]]`) y la región de la imagen `[x0, y0, x1, y1]` normalizada donde se la
ve. La zona `lobby` y el nodo `N0001` coinciden con `configs/location.toml` (hay un
test que lo comprueba). La cámara `laptop-webcam` es un placeholder. `enter_dbm` y
`hysteresis_db` son los valores por defecto del receptor; `calibrated = false` hasta medir.

## Procedimiento de calibración en sitio

Hacer con el tag en la posición real de uso (en el cuerpo/bolsillo, mismo
`TX_POWER_LEVEL`) y el PC donde quedará fijo. Registrar cada medida; no reutilizar
valores de otro lugar.

1. Alimenta el tag y arranca el receptor con `--jsonl > medida.jsonl` (o la tabla).
2. **Dentro**: párate 30 s en tres puntos dentro del área. Anota media, mínimo y p10 del RSSI.
3. **Borde**: en el límite del área, 30 s. Anota la media.
4. **Fuera**: 30 s a 2-3 m de distancia del borde, y detrás de una pared si aplica.
5. Elige `enter_dbm` entre la media del borde y el p10 de "dentro". Elige
   `hysteresis_db` ≥ la desviación típica del RSSI en el borde (típicamente 4-8 dB).
6. Camina de dentro a fuera y viceversa 5 veces: no debe parpadear.
7. Escribe en `configs/site_map.toml` los valores elegidos, pon `calibrated = true` y
   anota fecha, posiciones y estadísticas en un informe en `docs/reports/`.

### Notas de la prueba en vivo (sin calibrar)

Receptor del PC contra el C6 conectado por USB, 45 s, sin mover nada: 20 observaciones
aceptadas, 73 rechazadas como `duplicate` (reentregas del adaptador) y 30 anuncios de
otros dispositivos ignorados. RSSI crudo entre -86 y -71 dBm; confianza entre 0.20 y
0.65; zona `lobby`. Con el umbral provisional de -70 dBm todo quedó `outside`. Distancia
y orientación no se midieron, así que no son datos de calibración.

## Pendiente

- Calibrar con el procedimiento anterior.
- Medición de batería: el firmware anuncia `0xFF` hasta definir el pin ADC.
- Repositorio de personal persistente y clave de seudonimización de despliegue (#16).
- Sin autenticación del tag: la mitigación es un HMAC por tag (#16).
