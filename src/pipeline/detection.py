"""Conecta detectores del repo (`Protocol` `Detector`) al pipeline compartido.

`DetectorProcessor` es el `processor(stream_id, frame)` de `LiveVideoPipeline`:

- una **única instancia** de cada detector atiende a todos los streams, igual que
  el engine compartido de la arquitectura (nunca un modelo por cámara);
- convierte la salida de cada detector a primitivas JSON finitas (sin numpy ni
  arrays), que es lo único que acepta el bus estricto;
- gestiona el ciclo de vida: `warmup` en el primer frame y `close` al terminar.

Los frames siguen sin salir del plano de video: el detector recibe el array del
`Frame` y aquí solo se devuelve metadata.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .sources import Frame


def _num(value: Any, digits: int) -> float:
    v = float(value)  # acepta escalares numpy y los convierte a float de Python
    if not math.isfinite(v):
        raise ValueError(f"Valor no finito en una detección: {value!r}")
    return round(v, digits)


def to_json_detection(det: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Una detección del `Detector` → dict de primitivas JSON para el bus."""
    xyxy = det.get("xyxy")
    if xyxy is None or len(xyxy) != 4:
        raise ValueError(f"Detección sin caja xyxy válida: {det!r}")
    out: dict[str, Any] = {
        "model": model,
        "cls": int(det["cls"]),
        "conf": _num(det["conf"], 4),
        "xyxy": [_num(v, 1) for v in xyxy],
    }
    label = det.get("label")
    if label is not None:
        out["label"] = str(label)
    return out


class DetectorProcessor:
    """`processor` de `LiveVideoPipeline` que ejecuta uno o varios detectores."""

    def __init__(self, detectors: Sequence[Any], warmup: int = 3) -> None:
        if not detectors:
            raise ValueError("Se necesita al menos un detector")
        self.detectors = list(detectors)
        self._warmup = warmup
        self._ready = False
        self._lock = threading.Lock()
        self.frames = 0
        self.infer_ms: list[float] = []

    def warmup(self, image: Any) -> None:
        """Carga y calienta los modelos antes de abrir las cámaras.

        Si no se llama, el primer frame paga la carga y el warmup, y mientras tanto
        el pipeline descarta los frames que siguen llegando.
        """
        with self._lock:
            self._ensure_ready(
                Frame(width=image.shape[1], height=image.shape[0], data=image)
            )

    def _ensure_ready(self, frame: Frame) -> None:
        if self._ready:
            return
        for det in self.detectors:
            for _ in range(self._warmup):
                det.infer(frame.data)  # warmup con la forma real de entrada
        self._ready = True

    def __call__(self, stream_id: str, frame: Frame) -> list[dict[str, Any]]:
        if frame.data is None:
            return []
        # El despachador del pipeline es un solo hilo; el lock protege llamadas externas.
        with self._lock:
            self._ensure_ready(frame)
            start = time.perf_counter()
            out: list[dict[str, Any]] = []
            for det in self.detectors:
                out += [to_json_detection(d, det.name) for d in det.infer(frame.data)]
            self.infer_ms.append((time.perf_counter() - start) * 1000.0)
            self.frames += 1
            return out

    def close(self) -> None:
        for det in self.detectors:
            close = getattr(det, "close", None)
            if callable(close):
                close()
