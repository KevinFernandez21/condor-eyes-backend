"""Detector de armas en dos etapas: personas en el frame completo y armas en sus recortes.

Un arma de 15 px en un frame 1080p se vuelve un objeto de 60–100 px dentro del
recorte ampliado de la persona que la lleva. Las detecciones del frame completo se
conservan (armas sin persona visible) y se fusionan con las de los recortes por NMS.
Cumple el `Protocol` `Detector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TwoStageConfig:
    weapon_model: str = "runs/firearm/yolov8n_firearm_v3/weights/best.pt"
    person_model: str = "weights/yolov8n.pt"
    person_imgsz: int = 1280
    person_conf: float = 0.3
    weapon_imgsz: int = 640
    crop_size: int = 384  # lado mayor del recorte ampliado de cada persona
    pad_x: float = (
        0.35  # margen lateral relativo al ancho de la persona (brazos, arma larga)
    )
    pad_y: float = 0.10
    max_people: int = 16
    full_frame: bool = True  # también busca armas en el frame completo
    min_conf: float = 0.05
    nms_iou: float = 0.5
    device: str = "0"
    half: bool = True


def _nms(
    boxes: list[tuple[float, list[float]]], iou: float
) -> list[tuple[float, list[float]]]:
    from .evaluate import _iou

    kept: list[tuple[float, list[float]]] = []
    for c, b in sorted(boxes, key=lambda x: -x[0]):
        if all(_iou(b, k[1]) <= iou for k in kept):
            kept.append((c, b))
    return kept


class TwoStageDetector:
    name = "firearm-two-stage"

    def __init__(self, config: TwoStageConfig | None = None) -> None:
        self.config = config or TwoStageConfig()
        self._person: Any = None
        self._weapon: Any = None
        self._weapon_cls: int | None = None

    def _require(self) -> None:
        if self._weapon is None:
            from ultralytics import YOLO

            self._person = YOLO(self.config.person_model, task="detect")
            self._weapon = YOLO(self.config.weapon_model, task="detect")
            names = self._weapon.names
            self._weapon_cls = next(i for i, n in names.items() if n == "weapon")

    def _kw(self, imgsz: int) -> dict[str, Any]:
        c = self.config
        return {
            "imgsz": imgsz,
            "device": c.device,
            "verbose": False,
            "quantize": 16 if c.half and c.device != "cpu" else None,
        }

    def people(self, frame: np.ndarray) -> list[list[float]]:
        self._require()
        r = self._person.predict(
            frame,
            classes=[0],
            conf=self.config.person_conf,
            **self._kw(self.config.person_imgsz),
        )[0]
        boxes = sorted(
            r.boxes.xyxy.tolist(), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])
        )
        return boxes[: self.config.max_people]

    def infer(self, frame: np.ndarray) -> list[dict]:
        import cv2

        self._require()
        c = self.config
        h, w = frame.shape[:2]
        found: list[tuple[float, list[float]]] = []
        if c.full_frame:
            r = self._weapon.predict(
                frame,
                classes=[self._weapon_cls],
                conf=c.min_conf,
                **self._kw(c.weapon_imgsz),
            )[0]
            found += [
                (float(s), [float(v) for v in b])
                for b, s in zip(
                    r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), strict=True
                )
            ]
        crops, origins = [], []
        persons = self.people(frame)
        for x1, y1, x2, y2 in persons:
            bw, bh = x2 - x1, y2 - y1
            cx1, cy1 = max(0, int(x1 - c.pad_x * bw)), max(0, int(y1 - c.pad_y * bh))
            cx2, cy2 = min(w, int(x2 + c.pad_x * bw)), min(h, int(y2 + c.pad_y * bh))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            crop = frame[cy1:cy2, cx1:cx2]
            s = c.crop_size / max(crop.shape[:2])
            crops.append(
                cv2.resize(
                    crop,
                    (
                        max(1, round(crop.shape[1] * s)),
                        max(1, round(crop.shape[0] * s)),
                    ),
                )
            )
            origins.append((cx1, cy1, s))
        if crops:
            results = self._weapon.predict(
                crops,
                classes=[self._weapon_cls],
                conf=c.min_conf,
                **self._kw(c.crop_size),
            )
            for r, (ox, oy, s) in zip(results, origins, strict=True):
                for b, sc in zip(
                    r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), strict=True
                ):
                    found.append(
                        (
                            float(sc),
                            [
                                b[0] / s + ox,
                                b[1] / s + oy,
                                b[2] / s + ox,
                                b[3] / s + oy,
                            ],
                        )
                    )
        weapons = _nms(found, c.nms_iou)
        out = [{"xyxy": b, "cls": 1, "conf": s, "label": "weapon"} for s, b in weapons]
        out += [{"xyxy": b, "cls": 0, "conf": 1.0, "label": "person"} for b in persons]
        return out

    def warmup(self, n: int = 3) -> None:
        blank = np.zeros((720, 1280, 3), dtype="uint8")
        for _ in range(n):
            self.infer(blank)

    def close(self) -> None:
        self._person = self._weapon = None
