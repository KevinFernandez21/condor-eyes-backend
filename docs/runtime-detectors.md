# Detectores YOLO a través del runtime multiagente (issue #39)

Revisión de cómo se prueban los modelos YOLO (armas #1/#25 y vigilancia #9) y
medición de los detectores ejecutados **por el runtime**: pipeline compartido (#30)
→ bus tipado → siete roles AgentScope (#29). Se comparan con la ejecución standalone.

## Alcance de la revisión

La issue pide revisar "el setup de AgentScope de Juan para probar los modelos". Ese
trabajo no existía como código: en `main` y en las ramas, a fecha 2026-10-02, ningún
módulo de modelos (`src/firearm`, `src/surveillance`, `src/compare`) usaba AgentScope.
Por eso se revisó la integración que corresponde a esta issue:

- lo que entró con #29 y #30 y toca a los modelos: rol `inference`, `MetadataPublisher`,
  `LiveVideoPipeline` y validación del bus;
- el adaptador nuevo de esta issue, `src/pipeline/detection.py`.

## Cómo se conectan los detectores

```
VideoFileSource ──► LiveVideoPipeline ──processor──► DetectorProcessor ──► [armas v3, vigilancia]
                         │                              (1 instancia por modelo,
                         │                               JSON estricto)
                         └─► FrameMetadata ──► MetadataPublisher ──► InMemoryHub ──► Topic.DETECTIONS
                                                                                 └─► AgentRuntime: tracker → event → …
```

- `DetectorProcessor` (`src/pipeline/detection.py`) es el `processor(stream_id, frame)`
  del pipeline:
  - una única instancia de cada `Detector` atiende a todos los streams;
  - la salida pasa a `{"model", "cls", "label", "conf", "xyxy"}` con floats e ints de
    Python, finitos, sin numpy;
  - `warmup(image)` carga los modelos antes de abrir las cámaras y `close()` los libera.
- Los detectores conservan el `Protocol` `Detector` y la importación diferida de
  ultralytics/torch/cv2. El adaptador solo los llama.

## Revisión: hallazgos y cambios requeridos

| # | Dónde | Hallazgo | Cambio requerido |
|---|---|---|---|
| 1 | `src/agents/inference_agent.py:10,13`, `src/agents/handlers.py:79-93`, `src/pipeline/publisher.py:31` | La ruta declara que `inference` "controla el engine compartido y publica detecciones", pero `InferenceHandler` solo cuenta streams activos. Las detecciones las publica el pipeline con `source="pipeline"`, fuera del control de `check_publish_allowed` (`src/agents/runtime.py:124`) | Elegir y documentar una opción: `MetadataPublisher(..., source="inference")`, o declarar `pipeline` como productor legítimo de `DETECTIONS` y ajustar la descripción del rol |
| 2 | `src/pipeline/shared_pipeline.py:44` (`_classify_uri`) | Solo acepta `rtsp://` y USB. **No hay fuente de video grabado**, así que los modelos no se pueden probar por la ruta oficial con un archivo. Aquí se usó una `SourceFactory` propia de evaluación (`scripts/runtime_detectors.py:VideoFileSource`) | Añadir un `SourceKind.FILE` solo para evaluación (o una `RecordedVideoSource` en `src/pipeline/sources.py`), con lectura sincronizada opcional para pruebas reproducibles |
| 3 | `src/pipeline/live_pipeline.py:386-403` (`_dispatch_round` / `_deliver`) | El `processor` recibe **un frame a la vez** en un único hilo despachador, por turnos. No hay lotes entre streams, y #23 midió el engine con lote = N. Con dos modelos (~30 ms por frame en el runtime) el total para todas las cámaras es ~33 FPS: 8 FPS por cámara con 4 cámaras | API de `processor` por lotes (todos los frames de una ronda), alineada con `nvstreammux` batch N |
| 4 | `src/pipeline/live_pipeline.py:101` (`processor: Processor`) | El `processor` es un callable sin ciclo de vida. Si el modelo carga y se calienta en el primer frame, la cámara sigue llegando y se descartan frames: **35 de 40** en la primera prueba | Hooks `start()`/`close()` para el processor, o exigir `warmup` antes de `pipeline.start()`, como hace ahora `DetectorProcessor.warmup` |
| 5 | `src/pipeline/metadata.py:39` (`detections: tuple[Mapping[str, Any], ...]`), `src/agents/handlers.py:96` | El contenido de cada detección no tiene esquema ni versión. `tracker` reenvía claves libres | Registrar `DETECTIONS` v1 con `register_payload_versions` (`src/bus/hub.py:151`) y el esquema `{model, cls, label, conf, xyxy}` de `to_json_detection` |
| 6 | Proceso único: pipeline + agentes | La inferencia (pre/posproceso de ultralytics en Python) se ralentiza en el mismo proceso. Medido en el video del pasillo, p50 de inferencia: 14 ms standalone, 18 ms con un hilo decodificando o con el pipeline sin agentes, 27–30 ms con bus + 7 agentes. El bus solo cuesta ~1 ms | Ejecutar el plano de video e inferencia y el plano de agentes en **procesos separados** (en la Jetson, DeepStream/TensorRT fuera del GIL) y comunicarlos solo por metadata. Medirlo en la Jetson (#38, ARCH-009) |
| 7 | `src/pipeline/metadata.py:56`, `src/bus/hub.py` (`validate_payload`) | ✅ La validación estricta funciona: por el bus solo viajan primitivas JSON, y un array o un escalar numpy se rechaza. Comprobado en `tests/test_pipeline_detection.py` | Ninguno |
| 8 | Modelo de armas v3 (#25) | En MOT16-09 (personas, sin armas) marcó **3 detecciones `weapon`** en 525 frames: falsas alarmas en una escena concurrida | Añadir escenas concurridas sin armas a los negativos de la próxima versión; las alertas deben exigir persistencia temporal |

## Resultados: standalone frente a runtime

- **Hardware:** laptop Intel Core Ultra 9 275HX (24 hilos), 32 GB, NVIDIA GeForce RTX
  5080 Laptop GPU (CUDA, torch 2.14.1+cu130, ultralytics 8.4.170, FP16), Windows 11,
  Python 3.11.16. Es el dispositivo por defecto de las configuraciones de los
  detectores (`device="0"`), no el proxy sin CUDA de #23.
- **Contención:** dos entrenamientos de MATLAB ajenos ocupaban la CPU (~80 % del sistema).
- **Detectores:** armas v3 (`configs/firearm.toml`, 640, `conf=0.475`) + vigilancia v0
  (`configs/surveillance.toml`, 640, umbrales por clase). Son **dos modelos** en un solo
  processor, la excepción temporal de ARCH-006.
- **Modos:**
  - *standalone*: OpenCV + `infer()`;
  - *lockstep*: runtime completo, la fuente espera a que cada frame llegue al bus;
  - *tiempo real*: runtime completo, cámara simulada a 30 FPS.

| Video | Frames | Detecciones idénticas (lockstep / tiempo real) | Diferencia máx. | Descartados | Errores |
|---|---|---|---|---|---|
| `evaluation.mp4` (pasillo con rifle, 464×688) | 145 | **145 / 145** | 0 | 0 | 0 |
| MOT16-09 (1080p, multitud) | 525 | **525 / 525** | 0 | 0 | 0 |

Detecciones por clase, iguales en los tres modos:

| Video | Armas v3 | Vigilancia |
|---|---|---|
| Pasillo | 45 `weapon`, 145 `person` | 145 `person`, 25 `handbag`, 2 `backpack` |
| MOT16-09 | 3 `weapon` (falsas), 3320 `person` | 5443 `person`, 1240 `handbag`, 125 `backpack`, 7 `suitcase`, 2 `bicycle`, 1 `car` |

En todos los casos, `tracker` y `event` procesaron todos los mensajes sin fallos.

Latencia por frame, en ms. "Fuera de inferencia" = e2e − inferencia, frame a frame: es lo
que añade el camino pipeline → bus → agente.

| Video | Modo | Inferencia p50 / p90 | e2e p50 / p90 | Fuera de inferencia p50 / p90 |
|---|---|---|---|---|
| Pasillo | standalone | 14,1 / 32,0 | 14,4 / 32,4 | 0,3 / 0,5 |
| Pasillo | lockstep | 27,2 / 37,1 | 28,3 / 38,2 | **1,0 / 1,2** |
| Pasillo | tiempo real 30 FPS | 33,4 / 40,3 | 77,6 / 118,9 | 44,3 / 86,1 (cola) |
| MOT16-09 | standalone | 16,2 / 27,6 | 20,2 / 31,2 | 3,8 / 4,6 |
| MOT16-09 | lockstep | 29,2 / 37,7 | 34,5 / 43,0 | **5,0 / 5,7** |
| MOT16-09 | tiempo real 30 FPS | 30,2 / 38,6 | 53,7 / 106,0 | 21,6 / 73,6 (cola) |

- **Sobrecarga del runtime** sobre el camino del frame: **+0,6 ms** (pasillo) y
  **+1,2 ms** (MOT16-09) en p50.
- **En tiempo real**, la diferencia es espera en cola. Con dos modelos a ~30 ms por frame,
  una cámara a 30 FPS llega al límite: la cola acotada (4 frames) absorbe los picos y
  no hubo descartes en estos videos.
- **La inferencia** sube de ~14 a ~28 ms dentro del runtime por competencia de CPU y GIL
  en el mismo proceso (hallazgo 6).

Memoria del proceso (pico), en GB:

| Video | Modo | Working set | Privada |
|---|---|---|---|
| Pasillo | standalone | 1,80 | 3,31 |
| Pasillo | lockstep | 1,81 | 3,32 |
| Pasillo | tiempo real | 1,82 | 3,33 |
| MOT16-09 | standalone | 1,90 | 3,41 |
| MOT16-09 | lockstep | 1,90 | 3,42 |
| MOT16-09 | tiempo real | 1,92 | 3,43 |

El runtime (hub + 7 agentes + pipeline) añade **≤ 0,02 GB**. La RAM privada alta viene
de las DLL de torch+CUDA; ver #23 para el proxy sin torch.

## Un modelo o dos (entrada para #23)

Hoy corren **dos** modelos (armas v3 + vigilancia v0) en el mismo `DetectorProcessor`,
uno detrás de otro, con una instancia compartida por todos los streams. Es la excepción
temporal de ARCH-006, y su costo de memoria y FPS está en `docs/edge-budget.md`. Aquí,
en CUDA, los dos modelos suman ~14 ms de inferencia standalone (p50, pasillo); un YOLOv8n de armas solo (v1, misma arquitectura) tardó 8,5 ms (p50, `annotate`) en el mismo video.

## Reproducir

```bash
uv run python scripts/runtime_detectors.py all --video data/raw/cctv-weapon/evaluation.mp4
uv run python scripts/runtime_detectors.py all --video data/raw/mot/videos/MOT16-09.mp4
```

Resultados en `reports/runtime/<video>/{standalone,lockstep,realtime,comparison}.json`
(fuera de Git).
