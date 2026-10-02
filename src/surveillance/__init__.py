"""Detector general de personas y objetos para escenas CCTV (issue #9)."""

from __future__ import annotations

from .classes import NAMES
from .detector import SurveillanceConfig, SurveillanceDetector, load_surveillance_config

__all__ = [
    "NAMES",
    "SurveillanceConfig",
    "SurveillanceDetector",
    "load_surveillance_config",
]
