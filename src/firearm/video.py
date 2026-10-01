"""Anotación de video grabado: cajas, clase, confianza y FPS; mide latencia y memoria."""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

WEAPON_COLOR = (0, 0, 255)  # BGR rojo
OTHER_COLOR = (0, 200, 0)  # BGR verde


def latency_stats(lat_ms: list[float]) -> dict:
    """Media y percentiles p50/p90/p99 con el mismo criterio que `compare.benchmark`."""
    if not lat_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0}
    s = sorted(lat_ms)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return {"mean_ms": sum(s) / len(s), "p50_ms": p(0.50), "p90_ms": p(0.90), "p99_ms": p(0.99)}


def draw_detections(
    frame: Any,
    dets: list[dict],
    weapon_classes: Iterable[str] = ("handgun",),
    fps: float | None = None,
    infer_ms: float | None = None,
) -> Any:
    """Dibuja sobre `frame` (in place) cajas, `clase conf` y el overlay de FPS."""
    import cv2

    weapons = set(weapon_classes)
    for d in dets:
        x1, y1, x2, y2 = (round(v) for v in d["xyxy"])
        label = d.get("label", str(d["cls"]))
        color = WEAPON_COLOR if label in weapons else OTHER_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {d['conf']:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1, th + base + 2)
        cv2.rectangle(frame, (x1, ty - th - base - 2), (x1 + tw + 4, ty), color, -1)
        cv2.putText(frame, text, (x1 + 2, ty - base), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if fps is not None:
        hud = f"FPS {fps:5.1f}"
        if infer_ms is not None:
            hud += f" | infer {infer_ms:5.1f} ms"
        cv2.rectangle(frame, (8, 8), (16 + 13 * len(hud), 40), (0, 0, 0), -1)
        cv2.putText(frame, hud, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


class _MemoryProbe:
    """Pico de RSS del proceso (psutil) y de memoria CUDA reservada por torch."""

    def __init__(self) -> None:
        import psutil  # type: ignore[import-untyped]

        self._proc = psutil.Process()
        self.peak_rss = 0
        self._cuda = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._cuda = True
        except ImportError:
            pass

    def sample(self) -> None:
        self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)

    def result(self) -> dict:
        out: dict[str, Any] = {"peak_rss_mb": self.peak_rss / 2**20}
        if self._cuda:
            import torch

            out["peak_cuda_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
            out["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
            out["cuda_device"] = torch.cuda.get_device_name(0)
        return out


def annotate_video(
    detector: Any,
    input_path: str | Path,
    output_path: str | Path,
    weapon_classes: Iterable[str] | None = None,
    max_frames: int | None = None,
    warmup: int = 5,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Corre `detector` (batch 1) sobre un video y escribe un MP4 anotado.

    `weapon_classes` por defecto sale de `detector.config` (o `handgun`).
    Devuelve las métricas y además las guarda en `<output>.json`.
    """
    import cv2

    input_path, output_path = Path(input_path), Path(output_path)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {input_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames) if total > 0 else max_frames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter.fourcc(*"mp4v"), src_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo crear el video de salida: {output_path}")

    config = getattr(detector, "config", None)
    if weapon_classes is None:
        weapon_classes = getattr(config, "weapon_classes", None) or ("handgun",)
    weapons = tuple(weapon_classes)
    probe = _MemoryProbe()
    lat: list[float] = []
    weapon_frames = 0
    weapon_dets = 0
    ema_fps: float | None = None
    try:
        # Warmup con el primer frame real: la forma de entrada fija los kernels CUDA
        # y así la primera inferencia medida no carga ese costo.
        ok, pending = cap.read()
        for _ in range(warmup if ok else 0):
            detector.infer(pending.copy())
        t0 = last = time.perf_counter()
        while ok and (max_frames is None or len(lat) < max_frames):
            frame = pending
            s = time.perf_counter()
            dets = detector.infer(frame)
            lat.append((time.perf_counter() - s) * 1000.0)
            n_weapons = sum(1 for d in dets if d.get("label") in weapons)
            weapon_dets += n_weapons
            weapon_frames += n_weapons > 0
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            ema_fps = inst if ema_fps is None else 0.9 * ema_fps + 0.1 * inst
            last = now
            draw_detections(frame, dets, weapons, fps=ema_fps, infer_ms=lat[-1])
            writer.write(frame)
            probe.sample()
            if progress:
                progress(len(lat), total)
            ok, pending = cap.read()
    finally:
        cap.release()
        writer.release()
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    elapsed = time.perf_counter() - t0

    stats: dict[str, Any] = {
        "model": detector.name,
        "input": str(input_path),
        "output": str(output_path),
        "resolution": [width, height],
        "source_fps": src_fps,
        "frames": len(lat),
        "batch": 1,
        "avg_fps_end_to_end": len(lat) / elapsed if elapsed > 0 else 0.0,
        "avg_fps_inference": 1000.0 * len(lat) / sum(lat) if lat and sum(lat) > 0 else 0.0,
        **latency_stats(lat),
        "frames_with_weapon": weapon_frames,
        "weapon_detections": weapon_dets,
        **probe.result(),
    }
    if config is not None:
        stats["config"] = dict(vars(config))
    output_path.with_suffix(".json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats
