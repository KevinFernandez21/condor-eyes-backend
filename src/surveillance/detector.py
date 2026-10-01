"""Detector de personas y objetos para CCTV, compatible con el `Protocol` `Detector`."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .classes import COCO80_TO_IDX, NAMES


@dataclass
class SurveillanceConfig:
    # `version` identifica la combinación modelo + umbrales que consume el pipeline.
    version: str = "v0-coco-baseline"
    model: str = "weights/yolov8n.pt"
    imgsz: int = 640
    device: str = "0"
    half: bool = True
    iou: float = 0.6
    # Umbral por clase (calibrado en val); `default_conf` para las que no figuren.
    default_conf: float = 0.3
    conf: dict[str, float] = field(default_factory=dict)

    @property
    def precision(self) -> int | None:
        return 16 if self.half and self.device != "cpu" else None

    def threshold(self, label: str) -> float:
        return self.conf.get(label, self.default_conf)


def load_surveillance_config(
    path: str | Path | None = None, **overrides: Any
) -> SurveillanceConfig:
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        data = dict(raw.get("detector", {}))
        if "conf" in raw:
            data["conf"] = dict(raw["conf"])
    known = {f.name for f in fields(SurveillanceConfig)}
    if unknown := set(data) - known:
        raise ValueError(f"Claves desconocidas en [detector]: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    cfg = SurveillanceConfig(**data)
    if bad := set(cfg.conf) - set(NAMES):
        raise ValueError(f"Clases desconocidas en [conf]: {sorted(bad)}")
    return cfg


class SurveillanceDetector:
    """YOLOv8n (COCO de 80 clases o afinado a las 9 clases) con umbral por clase.

    Devuelve dicts `{"xyxy", "cls", "conf", "label"}` con `cls` en el espacio de
    `NAMES`, el mismo contrato que el resto de detectores del repo.
    """

    name = "yolov8n-surveillance"

    def __init__(
        self, config: SurveillanceConfig | None = None, min_conf: float | None = None
    ) -> None:
        self.config = config or SurveillanceConfig()
        # Para evaluación se baja el umbral y se aplica después.
        self.min_conf = min_conf
        self._model: Any = None
        self._map: dict[int, int] = {}

    def _require(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            model = YOLO(self.config.model, task="detect")
            names: dict[int, str] = model.names
            if len(names) == 80:
                self._map = dict(COCO80_TO_IDX)
            else:
                idx = {n: i for i, n in enumerate(NAMES)}
                self._map = {k: idx[v] for k, v in names.items() if v in idx}
            if not self._map:
                raise ValueError(f"El modelo no tiene ninguna de las clases {NAMES}")
            self._model = model
        return self._model

    def raw(self, frames: Any) -> list[list[tuple[int, float, list[float]]]]:
        """Predicción (lista de frames) → por frame, `(cls, conf, xyxy)` en el espacio de `NAMES`."""
        cfg = self.config
        model = self._require()
        conf = (
            self.min_conf
            if self.min_conf is not None
            else min([cfg.default_conf, *cfg.conf.values()])
        )
        results = model.predict(
            frames,
            imgsz=cfg.imgsz,
            conf=conf,
            iou=cfg.iou,
            device=cfg.device,
            quantize=cfg.precision,
            classes=sorted(self._map),
            verbose=False,
        )
        out = []
        for r in results:
            b = r.boxes
            out.append(
                [
                    (self._map[int(c)], float(s), [float(v) for v in box])
                    for box, c, s in zip(
                        b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist(), strict=True
                    )
                    if int(c) in self._map
                ]
            )
        return out

    def warmup(self, n: int = 5) -> None:
        import numpy as np

        blank = np.zeros((480, 640, 3), dtype="uint8")
        for _ in range(n):
            self.raw(blank)

    def infer(self, frame: Any) -> list[dict]:
        kept = sorted(
            (
                x
                for x in self.raw(frame)[0]
                if x[1] >= self.config.threshold(NAMES[x[0]])
            ),
            key=lambda x: -x[1],
        )
        return [
            {"xyxy": box, "cls": c, "conf": s, "label": NAMES[c]} for c, s, box in kept
        ]

    def close(self) -> None:
        self._model = None
