# Informe: re-identificación entre cámaras (issue #11)

Corrida del 2026-10-01 en la laptop (RTX 5080 Laptop), con
`uv run python scripts/reid.py benchmark`. Datos: Market-1501 v15.09.15, solo para
investigación.

| Split | Identidades | Imágenes |
|---|---|---|
| train | 676 | 11.674 |
| val (calibración) | 75 | 329 consultas / 933 galería |
| test (held-out) | 750 | 3.368 consultas / 15.913 galería, con distractores |

Las identidades no se solapan entre splits.

## Recuperación (protocolo Market-1501, test)

| Modelo | rank-1 | rank-5 | mAP | Dim. | Pesos FP16 |
|---|---|---|---|---|---|
| `color-hsv` (no biométrico) | 0,152 | 0,281 | 0,052 | 256 | — |
| `mobilenet_v3_small-imagenet` | 0,144 | 0,283 | 0,045 | 576 | 1,8 MB |
| `resnet18-imagenet` | 0,124 | 0,244 | 0,038 | 512 | 21,3 MB |
| `mobilenet_v3_small-reid` | 0,866 | 0,958 | 0,682 | 576 | 1,8 MB |
| **`resnet18-reid`** | **0,886** | 0,958 | **0,707** | 512 | 21,3 MB |

En val, los modelos re-ID dan rank-1 0,94–0,95 y mAP 0,86–0,87, y los demás 0,25–0,30
y 0,13–0,18.

Las CNN de ImageNet **no superan** al histograma de color: sin entrenamiento re-ID, la
apariencia genérica no distingue personas. El fine-tuning con identidades multiplica el
rank-1 por seis.

## Asociación con estado inconcluso (test, umbrales calibrados en val)

El umbral y el margen se eligieron en val para maximizar los enlaces correctos con
**≤ 1 % de falsos enlaces**. Todos los modelos cumplían ese objetivo en val.

| Modelo | Umbral / margen | Enlace correcto | Falso enlace (presente) | **Enlace perdido** | Falso enlace (ausente) | **Falso enlace total** |
|---|---|---|---|---|---|---|
| `color-hsv` | 0,875 / 0,05 | 1,8 % | 0,1 % | 98,0 % | 0,2 % | 0,2 % |
| `mobilenet_v3_small-imagenet` | 0,30 / 0,05 | 0,5 % | 0,0 % | 99,5 % | 0,0 % | 0,0 % |
| `resnet18-imagenet` | 0,925 / 0 | 2,3 % | 4,2 % | 93,6 % | 4,6 % | 4,4 % |
| `mobilenet_v3_small-reid` | 0,60 / 0 | **67,0 %** | 2,2 % | 30,8 % | 9,2 % | 5,7 % |
| **`resnet18-reid`** | 0,65 / 0 | 52,1 % | **1,2 %** | 46,7 % | **4,4 %** | **2,8 %** |

- *Presente:* la persona sí está en otra cámara de la galería.
- *Ausente:* se quitan sus tracks, así que lo correcto es `inconclusive` o `no_candidate`
  y cualquier enlace es falso.

Los falsos enlaces se cuentan **aparte** de los perdidos.

**El objetivo del 1 % de val no se sostuvo en test**: los falsos enlaces quedan entre
2,8 % y 5,7 %. Val tiene solo 75 identidades y una galería pequeña. En test hay 10
veces más personas y distractores, así que la probabilidad de un "parecido" por encima
del umbral crece. El peor caso es el escenario *ausente*: cuando la persona no está, el
modelo encuentra a alguien similar el 4–9 % de las veces. Hay que calibrar con una
galería del tamaño real del sitio y, para ese objetivo, subir el umbral.

## Coste

| Modelo | Batch 1 p50 / p90 | Lote de 16 p50 | Pico CUDA |
|---|---|---|---|
| `color-hsv` | 0,10 / 0,11 ms (CPU) | 1,4 ms | 0 |
| `mobilenet_v3_small-reid` | 13,1 / 14,5 ms | 12,8 ms | 170 MB |
| `resnet18-reid` | 6,0 / 7,6 ms | 6,1 ms | 358 MB |

- Las latencias se midieron **mientras otro entrenamiento compartía la GPU**, así que
  son cotas superiores. La de MobileNet, en particular, es anómala.
- El pico CUDA corresponde a lotes de 128 recortes durante el benchmark.
- En la Jetson, con lotes pequeños, el modelo cabe con holgura junto al detector
  compartido. Los pesos FP16 ocupan entre 2 y 21 MB, pero hay que medirlo allí.

## Decisión

1. **Embedding:** `resnet18-reid`. Es el que tiene menos falsos enlaces y el mejor mAP.
   `mobilenet_v3_small-reid` queda como alternativa si la memoria de la Jetson aprieta:
   enlaza más, pero con el doble de falsos enlaces.
2. **Estado inconcluso obligatorio:** incluso con el mejor modelo, el 47 % de las
   consultas queda inconclusa. Es preferible a forzar enlaces; el tracking continuo
   debe tolerar huecos.
3. **Filtro por tiempo y zona:** el baseline no biométrico solo de color es seguro pero
   casi inútil (1,8 % de enlaces). Debe ir **junto** al embedding, con la tabla de
   transiciones (`Transition`) que limita qué cámaras y zonas se conectan y en qué
   tiempo. Esto reduce los candidatos y con ello los falsos enlaces. Market no tiene
   tiempos sincronizados, así que este efecto no se pudo medir; solo está cubierto por
   tests unitarios.
4. **Recalibrar en el sitio:** antes de producción se calibra el umbral con grabaciones
   locales y una galería del tamaño real, con un objetivo de falsos enlaces ≤ 1 %
   medido en un held-out.
5. **Licencia:** los pesos `*-reid` derivan de Market-1501, que es **solo para
   investigación**. Para un uso no investigativo hay que reentrenar con datos propios o
   con licencia adecuada.

## Riesgos de privacidad

Ver `docs/reid-evaluation.md`: retención de 10 min, sin imágenes, alcance local, sin
identificación masiva y con auditoría de los usos fuera del tracking.
