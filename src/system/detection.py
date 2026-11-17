"""Cableado del detector del ejecutor, aislado en un único módulo.

Seguimiento tras el merge de #40: ``pipeline.detection.DetectorProcessor``
cubre lo mismo (un detector compartido como ``processor`` del pipeline, salida
JSON estricta, warmup antes de las cámaras). Entonces basta con sustituir
``RunnerDetector`` por ese adaptador en ``SystemApp`` (un solo punto:
``self._detection``), conservando la política de aquí: **una detección
malformada se descarta y se cuenta** (``malformed_dropped``), sin tumbar el
frame.

El detector corre del lado del pipeline: del frame solo salen detecciones.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from compare import FakeDetector
from compare.detectors import Detector
from pipeline import Frame

from .config import DetectorConfig
from .simulators import MovingDetector

logger = logging.getLogger(__name__)


def clean_detections(raw: Any) -> tuple[list[dict[str, Any]], int]:
    """Primitivas JSON finitas (sin numpy); devuelve (válidas, cuántas se descartaron)."""
    out: list[dict[str, Any]] = []
    dropped = 0
    for det in raw or []:
        try:
            box = [float(v) for v in det["xyxy"]]
            item: dict[str, Any] = {
                "xyxy": box,
                "cls": int(det["cls"]),
                "conf": float(det["conf"]),
            }
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if len(box) != 4 or not all(math.isfinite(v) for v in (*box, item["conf"])):
            dropped += 1
            continue
        for key in ("label", "stream_id", "frame_ts"):
            if isinstance(det.get(key), str):
                item[key] = det[key]
        track_id = det.get("track_id")
        if isinstance(track_id, int) and not isinstance(track_id, bool):
            item["track_id"] = track_id
        out.append(item)
    return out, dropped


def weights_path(cfg: DetectorConfig) -> str:
    """Prioridad: ``CONDOR_WEIGHTS`` > ``[detector].weights`` > surveillance.toml."""
    env = os.environ.get("CONDOR_WEIGHTS", "").strip()
    if env:
        return env
    if cfg.weights:
        return cfg.weights
    from surveillance.detector import load_surveillance_config

    weights = load_surveillance_config(cfg.config_path).model
    if not Path(weights).is_absolute():
        weights = str(Path(cfg.config_path).resolve().parent.parent / weights)
    return weights


def build_yolo(cfg: DetectorConfig) -> Detector:
    weights = weights_path(cfg)
    if not Path(weights).is_file():
        raise FileNotFoundError(
            f"Pesos de YOLOv8n no encontrados: {weights}. Colóquelos en esa ruta, "
            "o indique otra con la variable de entorno CONDOR_WEIGHTS o con "
            "[detector].weights en configs/system.toml. Se obtienen del asset "
            "oficial 'yolov8n.pt' de ultralytics; no se descarga nada automáticamente."
        )
    from surveillance.detector import SurveillanceDetector, load_surveillance_config

    device = "0"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    return SurveillanceDetector(
        load_surveillance_config(cfg.config_path, model=weights, device=device)
    )


def build_detector(cfg: DetectorConfig) -> Detector | None:
    if cfg.kind == "none":
        return None
    if cfg.kind == "fake":
        return FakeDetector(latency_ms=1.0)
    if cfg.kind == "moving":
        return MovingDetector()
    if cfg.kind == "scenes":
        raise ValueError(
            'El detector "scenes" requiere camera.kind = "fake" y [[cameras]]'
        )
    return build_yolo(cfg)


class RunnerDetector:
    """Un detector compartido: ``prepare()`` -> ``processor`` -> ``close()``."""

    def __init__(self, cfg: DetectorConfig, injected: Detector | None = None) -> None:
        self._cfg = cfg
        self._detector = injected
        self._injected = injected is not None
        self._lock = threading.Lock()  # infer en curso vs close
        self._closing = False
        self._state: dict[str, Any] = {
            "status": "starting",
            "detail": "",
            "name": None,
            "errors": 0,
            "malformed_dropped": 0,
        }

    def health(self) -> dict[str, Any]:
        return dict(self._state)

    def _degrade(self, exc: Exception) -> None:
        self._state.update(
            status="degraded",
            detail=f"{type(exc).__name__}: {exc}",
            errors=self._state["errors"] + 1,
        )

    def prepare(self) -> None:
        """Construye y calienta el detector (bloqueante; corre en un hilo)."""
        try:
            if not self._injected:
                self._detector = build_detector(self._cfg)
            if self._detector is None:
                self._state.update(status="disabled", detail="sin detector", name=None)
                return
            self._state["name"] = self._detector.name
            self._detector.warmup(1)
            self._state.update(status="ok", detail=self._detector.name)
        except Exception as exc:  # noqa: BLE001 - sin detector el sistema sigue
            self._degrade(exc)
            self.close()
            logger.warning("Detector degradado: %s", self._state["detail"])

    def close(self) -> None:
        """Cierra el detector esperando a que termine la inferencia en curso."""
        self._closing = True
        with self._lock:
            detector, self._detector = self._detector, None
        if detector is not None:
            with suppress(Exception):
                detector.close()

    def processor(self, stream_id: str, frame: Frame) -> list[Mapping[str, Any]] | None:
        """``Processor`` del pipeline: del frame solo salen detecciones."""
        if self._closing:
            return None
        with self._lock:
            detector = self._detector
            if detector is None or self._closing:
                return None
            try:
                infer_stream = getattr(detector, "infer_stream", None)
                raw = (
                    infer_stream(stream_id, frame.data)
                    if infer_stream is not None
                    else detector.infer(frame.data)
                )
            except Exception as exc:  # noqa: BLE001 - un frame malo no tumba el pipeline
                self._degrade(exc)
                return None
        try:
            detections, dropped = clean_detections(raw)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
            return None
        if dropped:
            self._state["malformed_dropped"] += dropped
        if self._state["status"] == "degraded":
            self._state.update(status="ok", detail=detector.name)
        return list(detections)
