"""Detector de armas compatible con el `Protocol` `Detector` de `compare`."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .config import FirearmConfig


def boxes_to_dets(
    xyxy: Iterable[Sequence[float]],
    cls: Iterable[float],
    conf: Iterable[float],
    names: Mapping[int, str],
    min_conf: float = 0.0,
    keep: Iterable[str] | None = None,
) -> list[dict]:
    """Convierte las salidas crudas del modelo a dicts y filtra por confianza y clase.

    `keep` vacío o None conserva todas las clases. Devuelve las detecciones
    ordenadas por confianza descendente.
    """
    keep_set = set(keep or ())
    dets: list[dict] = []
    for box, c, s in zip(xyxy, cls, conf, strict=True):
        cid = int(c)
        label = names.get(cid, str(cid))
        score = float(s)
        if score < min_conf or (keep_set and label not in keep_set):
            continue
        dets.append({"xyxy": [float(v) for v in box], "cls": cid, "conf": score, "label": label})
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


class FirearmDetector:
    """YOLOv8n afinado para `person`/`handgun`. Batch 1, un frame BGR por llamada."""

    name = "yolov8n-firearm"

    def __init__(self, config: FirearmConfig | None = None) -> None:
        self.config = config or FirearmConfig()
        self._model: Any = None
        self._class_ids: list[int] | None = None

    def _require(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ImportError("Falta 'ultralytics'. Ejecuta: uv sync") from e
            self._model = YOLO(self.config.model, task="detect")
            names: dict[int, str] = self._model.names
            wanted = set(self.config.classes)
            missing = wanted - set(names.values())
            if missing:
                raise ValueError(
                    f"El modelo no tiene las clases {sorted(missing)}; tiene {sorted(names.values())}"
                )
            self._class_ids = [i for i, n in names.items() if n in wanted] or None
        return self._model

    @property
    def names(self) -> dict[int, str]:
        return dict(self._require().names)

    def _predict(self, frame: Any) -> Any:
        cfg = self.config
        return self._require().predict(
            frame,
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            quantize=cfg.precision,
            classes=self._class_ids,
            verbose=False,
        )[0]

    def warmup(self, n: int = 5) -> None:
        import numpy as np

        blank = np.zeros((480, 640, 3), dtype="uint8")
        for _ in range(n):
            self._predict(blank)

    def infer(self, frame: Any) -> list[dict]:
        out = self._predict(frame)
        boxes = getattr(out, "boxes", None)
        if boxes is None:
            return []
        return boxes_to_dets(
            boxes.xyxy.tolist(),
            boxes.cls.tolist(),
            boxes.conf.tolist(),
            out.names,
            min_conf=self.config.conf,
            keep=self.config.classes,
        )

    def close(self) -> None:
        self._model = None
        self._class_ids = None
