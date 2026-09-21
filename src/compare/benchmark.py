"""Harness de comparación pensado para un .ipynb del usuario."""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable
from typing import Any


def _frames_from(source: Any, limit: int) -> Iterable[Any]:
    if callable(source):
        for _ in range(limit):
            yield source()
        return
    if isinstance(source, (list, tuple)):
        yield from list(source)[:limit]
        return
    if source is None:
        for _ in range(limit):
            yield None
        return
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore
    if isinstance(source, str):
        import cv2

        cap = cv2.VideoCapture(source)
        n = 0
        try:
            while n < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
                n += 1
        finally:
            cap.release()
        return
    if np is not None:
        blank = np.zeros((480, 640, 3), dtype="uint8")
        for _ in range(limit):
            yield blank
        return
    raise TypeError("source debe ser callable, lista de frames o ruta de video")


def benchmark_detector(
    detector: Any,
    source: Any = None,
    frames: int = 100,
    warmup: int = 5,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Corre un detector sobre N frames y devuelve latencias y FPS."""
    detector.warmup(warmup)
    lat: list[float] = []
    counts: list[int] = []
    t0 = time.perf_counter()
    for i, frame in enumerate(_frames_from(source, frames)):
        s = time.perf_counter()
        dets = detector.infer(frame)
        lat.append((time.perf_counter() - s) * 1000.0)
        counts.append(len(dets))
        if progress:
            progress(i + 1, frames)
    total = time.perf_counter() - t0
    lat_sorted = sorted(lat)
    p = lambda q: lat_sorted[min(len(lat_sorted) - 1, int(q * len(lat_sorted)))] if lat_sorted else 0.0
    return {
        "model": detector.name,
        "frames": len(lat),
        "fps": (len(lat) / total) if total > 0 else 0.0,
        "mean_ms": statistics.fmean(lat) if lat else 0.0,
        "p50_ms": p(0.50),
        "p90_ms": p(0.90),
        "p99_ms": p(0.99),
        "mean_dets": statistics.fmean(counts) if counts else 0.0,
    }


def compare_models(detectors: list[Any], **kwargs: Any) -> list[dict]:
    """Compara varios detectores sobre la misma fuente. Devuelve una fila por modelo."""
    rows: list[dict] = []
    for det in detectors:
        try:
            rows.append(benchmark_detector(det, **kwargs))
        finally:
            close = getattr(det, "close", None)
            if callable(close):
                close()
    return rows
