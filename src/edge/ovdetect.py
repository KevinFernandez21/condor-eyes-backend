"""Detector YOLOv8 sobre OpenVINO puro, sin torch ni ultralytics.

Es el runtime "lean" del presupuesto: el proceso de producción no necesita torch
(en la Jetson será TensorRT/DeepStream), y en Windows las DLL de torch+CUDA
reservan más de 1 GB aunque no se usen. Preproceso (letterbox) y NMS replican los
de ultralytics; `tests/test_edge.py` compara ambas salidas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensiona manteniendo aspecto y rellena con gris 114, como ultralytics."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[py : py + nh, px : px + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return out, r, (px, py)


class OVDetector:
    """YOLOv8 exportado a OpenVINO (carpeta `*_openvino_model` con batch dinámico)."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "GPU.0",
        imgsz: int = 640,
        conf: float = 0.3,
        iou: float = 0.6,
    ) -> None:
        import openvino as ov  # type: ignore[import-untyped]

        model_dir = Path(model_dir)
        xml = next(model_dir.glob("*.xml"))
        self.compiled: Any = ov.Core().compile_model(
            str(xml), device, {"PERFORMANCE_HINT": "LATENCY"}
        )
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def predict(
        self, frames: list[np.ndarray]
    ) -> list[list[tuple[int, float, list[float]]]]:
        """Lote de frames BGR → por frame, `(cls, conf, xyxy)` en píxeles del frame."""
        import cv2

        boxes = [letterbox(f, self.imgsz) for f in frames]
        batch = (
            np.stack([b[0][:, :, ::-1].transpose(2, 0, 1) for b in boxes]).astype(
                np.float32
            )
            / 255.0
        )
        out = self.compiled([batch])[0]  # (N, 4 + nc, anchors)
        results: list[list[tuple[int, float, list[float]]]] = []
        for k, (_, r, (px, py)) in enumerate(boxes):
            pred = out[k].T  # (anchors, 4 + nc)
            scores = pred[:, 4:]
            cls = scores.argmax(1)
            conf = scores[np.arange(len(cls)), cls]
            keep = conf >= self.conf
            if not keep.any():
                results.append([])
                continue
            xywh, cls, conf = pred[keep, :4], cls[keep], conf[keep]
            x1 = (xywh[:, 0] - xywh[:, 2] / 2 - px) / r
            y1 = (xywh[:, 1] - xywh[:, 3] / 2 - py) / r
            w, h = xywh[:, 2] / r, xywh[:, 3] / r
            kept = cv2.dnn.NMSBoxesBatched(
                np.stack([x1, y1, w, h], 1).tolist(),
                conf.tolist(),
                cls.tolist(),
                self.conf,
                self.iou,
            )
            idx = np.asarray(kept).flatten()
            results.append(
                [
                    (
                        int(cls[i]),
                        float(conf[i]),
                        [
                            float(x1[i]),
                            float(y1[i]),
                            float(x1[i] + w[i]),
                            float(y1[i] + h[i]),
                        ],
                    )
                    for i in idx
                ]
            )
        return results
