# Tag BLE con XIAO ESP32-C6 y receptor del PC

Estado: **firmware y receptor probados con hardware real; umbrales sin calibrar**.
Issue #26 (relacionado con #16, localización por zonas).

Con un solo radio no hay triangulación: lo medible es la **presencia dentro de un
área** (RSSI cerca de un receptor fijo). Fuera de alcance: multi-nodo, LoRaWAN, rostros.

## Topología

```
XIAO ESP32-C6 (tag, lo lleva la persona) ──BLE──▶ adaptador Bluetooth del PC (receptor fijo, junto a la cámara)
                                                       │ src/tagbridge: filtra el tag enrolado
                                                       ▼
                              observación payload v1 + estado inside/outside (umbral + histéresis)
```

El receptor del PC hace el papel del "nodo de zona" del contrato de #16 (`node = N0001`).
Más adelante puede sustituirse por un segundo ESP32 sin cambiar el contrato.

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
cuenta en `stats["ignored"]`. Firmware y build: `firmware/xiao_c6_tag/README.md`.

## Observación emitida (payload v1)

Igual que en `docs/location.md` de #16 (en la rama `feat/tag-zone-localization`):

```json
{"v":1,"ch":"ble","tag":"58E6C515220A","node":"N0001","rssi":-74,"ts":1790914468536,"seq":514207471}
```

`bat` se omite si el tag no la reporta. `ts` es la hora del PC en ms UTC. Los `seq`
repetidos u hasta 8 por detrás del último se descartan (Windows reentrega anuncios);
uno mucho más atrás se toma como reinicio del tag.

## Receptor en vivo

```bash
uv run python scripts/tag_receiver.py --tag 58E6C515220A            # tabla en vivo
uv run python scripts/tag_receiver.py --tag 58E6C515220A --jsonl    # observaciones v1
uv run python scripts/tag_receiver.py --tag ... --duration 30 --enter-dbm -65 --hysteresis-db 6
```

Salida: RSSI crudo, media móvil (`--window`, 5 por defecto), `inside`/`outside`,
confianza `[0, 1]` y `seq`. Entra cuando la media es `>= enter_dbm`; sale cuando es
`< enter_dbm - hysteresis_db`. Sin anuncios durante `--lost-after-s` pasa a `outside`
con confianza 0 (`reason = lost`). La confianza es el margen al umbral de decisión
entre 10 dB (heurística provisional). El ID del tag se pasa por argumento; no se
versiona.

## Site map

`configs/site_map.toml` une cada zona con su receptor (`[[receivers]]`), su cámara
(`[[cameras]]`) y la región de la imagen `[x0, y0, x1, y1]` normalizada donde se la
ve. La cámara `laptop-webcam` es un placeholder. `enter_dbm` y `hysteresis_db` de la
zona son los valores por defecto del receptor; `calibrated = false` hasta medir.

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

### Notas de la primera prueba (sin calibrar)

El receptor corrió contra el C6 conectado por USB al PC. El RSSI varió entre -74 y
-91 dBm, con huecos de varios segundos entre callbacks: el adaptador del PC entrega
una fracción de los anuncios. Distancia y orientación no se midieron, así que no son
datos de calibración.

## Pendiente

- Calibrar con el procedimiento anterior.
- Integrar el receptor con `LocationService` de `src/location` (#16) en lugar de la
  lógica de umbral local de `tagbridge.presence`.
- Medición de batería: el firmware anuncia `0xFF` hasta definir el pin ADC.
- Sin autenticación del tag: igual que en #16, la mitigación es un HMAC por tag.
