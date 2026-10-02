# Firmware del tag XIAO ESP32-C6

Anuncia por BLE como tag de personal (issue #26). Formato del anuncio y procedimiento
de calibración: `docs/xiao-c6-tag.md`.

## Requisitos

- [`arduino-cli`](https://arduino.github.io/arduino-cli/) y el core `esp32:esp32` (probado con 3.3.12).
- Placa: **Seeed Studio XIAO ESP32-C6**, FQBN `esp32:esp32:XIAO_ESP32C6`.
  `CDCOnBoot=cdc` activa `Serial` por el USB nativo.

```bash
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
```

## Compilar y flashear

```bash
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C6:CDCOnBoot=cdc firmware/xiao_c6_tag
arduino-cli upload -p COM8 --fqbn esp32:esp32:XIAO_ESP32C6:CDCOnBoot=cdc firmware/xiao_c6_tag
```

Cambia `COM8` por el puerto de tu placa (`arduino-cli board list`).

## Configuración

Se cambia con `--build-property "build.extra_flags=-DNOMBRE=valor"` en `compile`
(o editando las macros al inicio del `.ino`):

| Macro | Defecto | Efecto |
|-------|---------|--------|
| `ADV_INTERVAL_MS` | `100` | Periodo entre anuncios; cada periodo sube `seq` en 1. |
| `TAG_ID_OVERRIDE` | `""` | ID de 12 dígitos hex. Vacío = MAC Bluetooth del chip. |
| `BATTERY_ADC_PIN` | `-1` | Pin ADC de la batería; `-1` anuncia `0xFF` (desconocida). |
| `TX_POWER_LEVEL` | `ESP_PWR_LVL_P3` | Potencia de TX; recalibrar el RSSI si se cambia. |

## Log serie (115200)

Una línea por anuncio; al abrir el puerto la placa se reinicia y emite `BOOT`:

```
BOOT fw=1.0.0 tag=58E6C515220A name=CE-TAG-220A interval_ms=100
ADV tag=58E6C515220A seq=4225449557 bat=255
```

`seq` arranca en un valor aleatorio en cada arranque y sube de a 1; `bat=255` es
batería desconocida.
