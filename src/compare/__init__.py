"""Comparación YOLOv8n vs PeopleNet para llamar desde un .ipynb propio."""
from .benchmark import benchmark_detector, compare_models
from .detectors import FakeDetector, PeopleNetDetector, YOLOv8nDetector

__all__ = [
    "FakeDetector",
    "PeopleNetDetector",
    "YOLOv8nDetector",
    "benchmark_detector",
    "compare_models",
]
