"""Detección (YuNet) y embedding facial (SFace FP32/INT8 o baseline CNN genérica).

Modelos de OpenCV Zoo: YuNet (MIT) y SFace (Apache-2.0), ambos ONNX y ligeros
para edge. El baseline genérico (MobileNetV3 de ImageNet, sin entrenamiento
facial) sirve para mostrar cuánto aporta un modelo específico de caras.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "yunet": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface": f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "sface_int8": f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx",
}


def model_path(name: str, root: str | Path = "weights/faceid") -> Path:
    p = Path(root) / Path(MODELS[name]).name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(MODELS[name], tmp)
        tmp.replace(p)
    return p


@dataclass(frozen=True, slots=True)
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h
    landmarks: tuple[float, ...]  # 5 puntos (x, y): ojos, nariz, comisuras
    score: float
    raw: np.ndarray  # fila de YuNet (necesaria para alignCrop)

    @property
    def size(self) -> float:
        return min(self.box[2], self.box[3])

    @property
    def yaw_ratio(self) -> float:
        """Desplazamiento horizontal de la nariz respecto al centro de los ojos (≈ giro)."""
        rx, _, lx, _, nx = self.landmarks[:5]
        eye_dist = abs(lx - rx) or 1.0
        return (nx - (rx + lx) / 2) / eye_dist


class FaceEngine:
    def __init__(
        self,
        embedder: str = "sface",
        det_score: float = 0.7,
        upscale_to: int = 224,
        weights_root: str | Path = "weights/faceid",
    ) -> None:
        import cv2

        self.embedder_name = embedder
        self.upscale_to = upscale_to
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path("yunet", weights_root)), "", (320, 320), det_score, 0.3, 50
        )
        self._sface: Any = None
        self._cnn: Any = None
        if embedder in ("sface", "sface_int8"):
            self._sface = cv2.FaceRecognizerSF.create(
                str(model_path(embedder, weights_root)), ""
            )
        elif embedder == "mobilenet_imagenet":
            import torch
            import torchvision  # type: ignore[import-untyped]

            m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            m.classifier = torch.nn.Identity()
            self._cnn = m.eval()
        elif embedder != "detector_only":  # solo detecta; otro Embedder calcula el vector
            raise ValueError(f"Embedder desconocido: {embedder}")

    def _prepare(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Las caras recortadas pequeñas (112 px) se agrandan para que YuNet las detecte."""
        import cv2

        h, w = img.shape[:2]
        scale = self.upscale_to / max(h, w) if max(h, w) < self.upscale_to else 1.0
        if scale != 1.0:
            img = cv2.resize(
                img,
                (round(w * scale), round(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        return img, scale

    def detect(self, img: np.ndarray) -> tuple[np.ndarray, list[Face]]:
        img, _ = self._prepare(img)
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, rows = self.detector.detect(img)
        faces = [
            Face(
                (float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                tuple(float(v) for v in r[4:14]),
                float(r[14]),
                r,
            )
            for r in (rows if rows is not None else [])
        ]
        faces.sort(key=lambda f: -f.size)
        return img, faces

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        """`img` debe ser la imagen devuelta por `detect` (misma escala que `face`)."""
        if self._sface is None and self._cnn is None:
            raise RuntimeError("FaceEngine('detector_only') no calcula embeddings")
        if self._sface is not None:
            crop = self._sface.alignCrop(img, face.raw)
            v = self._sface.feature(crop).flatten().astype("float32")
        else:
            import cv2
            import torch

            x, y, w, h = (round(v) for v in face.box)
            crop = cv2.resize(img[max(0, y) : y + h, max(0, x) : x + w], (112, 112))
            t = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).float().div(255)
            t = (
                (t - torch.tensor([0.485, 0.456, 0.406]))
                / torch.tensor([0.229, 0.224, 0.225])
            ).permute(2, 0, 1)
            with torch.inference_mode():
                v = self._cnn(t[None])[0].numpy().astype("float32")
        return v / max(float(np.linalg.norm(v)), 1e-9)
