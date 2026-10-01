"""Prototipo de detección de armas de fuego en video grabado (issue #1)."""
from __future__ import annotations

from .config import FirearmConfig, load_config
from .detector import FirearmDetector, boxes_to_dets
from .video import annotate_video, draw_detections, latency_stats

__all__ = [
    "FirearmConfig",
    "FirearmDetector",
    "annotate_video",
    "boxes_to_dets",
    "draw_detections",
    "latency_stats",
    "load_config",
]
