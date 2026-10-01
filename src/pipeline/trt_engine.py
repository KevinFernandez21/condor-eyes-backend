"""Configuración del único engine TensorRT compartido."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TensorRTEngineSpec:
    """Referencia versionada a un engine FP16 no almacenado en Git."""

    path: Path
    version: str
    batch_size: int
    precision: str = "fp16"

    def __post_init__(self) -> None:
        if self.path.suffix != ".engine":
            raise ValueError("El artefacto TensorRT debe usar la extensión .engine")
        if self.precision.lower() != "fp16":
            raise ValueError("Solo se permite precisión FP16 en producción")
        if not 1 <= self.batch_size <= 8:
            raise ValueError("El batch debe estar entre 1 y 8 para la Orin Nano")
