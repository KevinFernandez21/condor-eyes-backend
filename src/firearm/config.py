"""Configuración del detector de armas: modelo, umbrales, tamaño de entrada y dispositivo."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class FirearmConfig:
    # .pt entrenado (laptop) o engine TensorRT FP16 (Jetson, a futuro).
    model: str = "runs/firearm/yolov8n_mgd_usrt/weights/best.pt"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    # "0" = primera GPU CUDA, "cpu" para forzar CPU.
    device: str = "0"
    # FP16 en GPU; se ignora en CPU.
    half: bool = True
    # Clases a conservar por nombre; vacío = todas las del modelo.
    classes: list[str] = field(default_factory=lambda: ["handgun", "person"])
    # Clases que se consideran arma (color de alerta y métricas de positivos).
    weapon_classes: list[str] = field(default_factory=lambda: ["handgun"])

    @property
    def precision(self) -> int | None:
        """Valor de `quantize` para ultralytics: 16 (FP16) en GPU, None (FP32) en CPU."""
        return 16 if self.half and self.device != "cpu" else None


def load_config(path: str | Path | None = None, **overrides: Any) -> FirearmConfig:
    """Carga la sección [detector] de un TOML y aplica los overrides no nulos."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f).get("detector", {})
    known = {f.name for f in fields(FirearmConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Claves desconocidas en [detector]: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    cfg = FirearmConfig(**data)
    if not 0.0 <= cfg.conf <= 1.0 or not 0.0 <= cfg.iou <= 1.0:
        raise ValueError("conf e iou deben estar en [0, 1]")
    if cfg.imgsz <= 0 or cfg.imgsz % 32:
        raise ValueError("imgsz debe ser múltiplo positivo de 32")
    return cfg
