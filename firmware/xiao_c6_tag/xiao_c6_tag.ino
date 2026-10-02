// Tag de personal para Condor Eyes: XIAO ESP32-C6 anunciando por BLE (issue #26).
//
// Datos de fabricante (company ID 0xFFFF), 14 bytes little-endian tras el company ID:
//   "CE" | versión=1 | tag[6] | seq u32 | batería u8 (0xFF = desconocida)
// Debe coincidir con src/tagbridge/protocol.py.
//
// Build:  arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C6:CDCOnBoot=cdc firmware/xiao_c6_tag
// Flash:  arduino-cli upload -p COM8 --fqbn esp32:esp32:XIAO_ESP32C6:CDCOnBoot=cdc firmware/xiao_c6_tag

#include <BLEDevice.h>
#include <BLEAdvertising.h>
#include <esp_mac.h>
#include <Preferences.h>

// --- Configuración --------------------------------------------------------------
#define FW_VERSION "1.0.0"

// Periodo entre anuncios (ms). Cada periodo se publica un seq nuevo.
#ifndef ADV_INTERVAL_MS
#define ADV_INTERVAL_MS 100
#endif

// ID del tag: 12 dígitos hex. Vacío = derivado de la MAC Bluetooth del chip.
#ifndef TAG_ID_OVERRIDE
#define TAG_ID_OVERRIDE ""
#endif

// Pin ADC para medir batería. -1 = sin sensor (se anuncia 0xFF = desconocida).
#ifndef BATTERY_ADC_PIN
#define BATTERY_ADC_PIN -1
#endif

// Potencia de TX del anuncio (la calibración de RSSI depende de este valor).
#ifndef TX_POWER_LEVEL
#define TX_POWER_LEVEL ESP_PWR_LVL_P3
#endif

// El seq debe ser monótono también entre reinicios: el anti-replay de LocationService
// (#16) rechaza un seq menor (o 2^31 mayor) que el último visto. Se reservan bloques en
// NVS: al arrancar se retoma desde el inicio del bloque guardado y se reserva el
// siguiente; el valor guardado siempre es mayor que cualquier seq ya emitido.
// Un reinicio salta como mucho SEQ_BLOCK; una escritura cada SEQ_BLOCK anuncios.
#ifndef SEQ_BLOCK
#define SEQ_BLOCK 1000
#endif

static const uint16_t COMPANY_ID = 0xFFFF;
static const uint8_t PAYLOAD_VERSION = 1;
static const uint8_t BATTERY_UNKNOWN = 0xFF;

static uint8_t tagId[6];
static uint32_t seq;
static uint32_t seqBlockEnd;
static Preferences prefs;
static BLEAdvertising *advertising;

static bool parseTagOverride(const char *hex, uint8_t *out) {
  if (strlen(hex) != 12) return false;
  for (int i = 0; i < 6; i++) {
    char pair[3] = {hex[2 * i], hex[2 * i + 1], 0};
    char *end;
    long v = strtol(pair, &end, 16);
    if (*end != 0) return false;
    out[i] = (uint8_t)v;
  }
  return true;
}

static uint8_t readBatteryPct() {
#if BATTERY_ADC_PIN >= 0
  // Divisor 2:1 supuesto: 3.0 V = 0 %, 4.2 V = 100 %. Ajustar al hardware real.
  float v = analogReadMilliVolts(BATTERY_ADC_PIN) * 2.0f / 1000.0f;
  float pct = (v - 3.0f) / 1.2f * 100.0f;
  return (uint8_t)constrain((int)pct, 0, 100);
#else
  return BATTERY_UNKNOWN;
#endif
}

static void publishAdvertisement(uint8_t battery) {
  uint8_t buf[2 + 14];
  buf[0] = COMPANY_ID & 0xFF;
  buf[1] = COMPANY_ID >> 8;
  buf[2] = 'C';
  buf[3] = 'E';
  buf[4] = PAYLOAD_VERSION;
  memcpy(&buf[5], tagId, 6);
  memcpy(&buf[11], &seq, 4);  // ESP32 es little-endian
  buf[15] = battery;

  String mfg;
  mfg.concat((const char *)buf, sizeof(buf));  // con longitud: seq/tag pueden contener 0x00

  BLEAdvertisementData data;
  data.setFlags(ESP_BLE_ADV_FLAG_GEN_DISC | ESP_BLE_ADV_FLAG_BREDR_NOT_SPT);
  data.setManufacturerData(mfg);
  advertising->setAdvertisementData(data);
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) delay(10);

  if (!(strlen(TAG_ID_OVERRIDE) && parseTagOverride(TAG_ID_OVERRIDE, tagId))) {
    esp_read_mac(tagId, ESP_MAC_BT);
  }
  prefs.begin("cetag", false);
  seq = prefs.getUInt("seqblk", 0);
  seqBlockEnd = seq + SEQ_BLOCK;
  prefs.putUInt("seqblk", seqBlockEnd);

  char name[16];
  snprintf(name, sizeof(name), "CE-TAG-%02X%02X", tagId[4], tagId[5]);
  BLEDevice::init(name);
  BLEDevice::setPower(TX_POWER_LEVEL);

  advertising = BLEDevice::getAdvertising();
  // Unidades de 0.625 ms; se fija min = max = intervalo configurado.
  uint16_t units = (uint16_t)(ADV_INTERVAL_MS / 0.625f);
  advertising->setMinInterval(units);
  advertising->setMaxInterval(units);
  advertising->setScanResponse(false);  // el anuncio ya lleva todo; sin respuesta de escaneo

  Serial.printf("BOOT fw=%s tag=%02X%02X%02X%02X%02X%02X name=%s interval_ms=%d\n", FW_VERSION,
                tagId[0], tagId[1], tagId[2], tagId[3], tagId[4], tagId[5], name,
                ADV_INTERVAL_MS, (unsigned long)seq);
}

void loop() {
  if (seq - (seqBlockEnd - SEQ_BLOCK) >= SEQ_BLOCK) {  // aritmética módulo 2^32
    seqBlockEnd += SEQ_BLOCK;
    prefs.putUInt("seqblk", seqBlockEnd);
  }
  uint8_t battery = readBatteryPct();
  publishAdvertisement(battery);
  advertising->start();
  Serial.printf("ADV tag=%02X%02X%02X%02X%02X%02X seq=%lu bat=%u\n", tagId[0], tagId[1], tagId[2],
                tagId[3], tagId[4], tagId[5], (unsigned long)seq, (unsigned)battery);
  delay(ADV_INTERVAL_MS);
  advertising->stop();
  seq++;
}
