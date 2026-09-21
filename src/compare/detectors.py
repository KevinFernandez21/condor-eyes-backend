"""Detectores con interfaz común para comparar en notebook."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Detector(Protocol):
    name: str

    def warmup(self, n: int = 5) -> None: ...
    def infer(self, frame: Any) -> list[dict]: ...
    def close(self) -> None: ...


class YOLOv8nDetector:
    """YOLOv8n vía ultralytics. `weights` admite .pt o engine TensorRT FP16."""

    name = "yolov8n"

    def __init__(self, weights: str | Path = "yolov8n.pt", imgsz: int = 640) -> None:
        self.weights = str(weights)
        self.imgsz = imgsz
        self._model: Any = None

    def _require(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ImportError(
                    "Falta 'ultralytics'. Instala con: pip install ultralytics"
                ) from e
            self._model = YOLO(self.weights)
        return self._model

    def warmup(self, n: int = 5) -> None:
        import numpy as np

        model = self._require()
        blank = np.zeros((480, 640, 3), dtype="uint8")
        for _ in range(n):
            model.predict(blank, imgsz=self.imgsz, verbose=False)

    def infer(self, frame: Any) -> list[dict]:
        model = self._require()
        out = model.predict(frame, imgsz=self.imgsz, verbose=False)[0]
        dets: list[dict] = []
        boxes = getattr(out, "boxes", None)
        if boxes is None:
            return dets
        for b, c, s in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            dets.append({"xyxy": b, "cls": int(c), "conf": float(s)})
        return dets

    def close(self) -> None:
        self._model = None


class PeopleNetDetector:
    """PeopleNet vía engine TensorRT (.engine) o fallback ONNX.

    En Jetson se usa el .engine FP16 generado con tao-converter/trtexec.
    Fuera de Jetson acepta .onnx vía onnxruntime para comparar lógica.
    """

    name = "peoplenet"

    def __init__(self, weights: str | Path, input_size: tuple[int, int] = (640, 640)) -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"Pesos no encontrados: {self.weights}")
        self.input_size = input_size
        self._backend: Any = None

    def _require(self) -> Any:
        suffix = self.weights.suffix.lower()
        if suffix == ".engine":
            try:
                import tensorrt as trt  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Falta 'tensorrt'. El .engine solo corre en Jetson."
                ) from e
            return ("trt", trt)
        if suffix == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise ImportError(
                    "Falta 'onnxruntime'. Instala con: pip install onnxruntime"
                ) from e
            if self._backend is None:
                self._backend = ort.InferenceSession(
                    str(self.weights), providers=["CPUExecutionProvider"]
                )
            return ("onnx", self._backend)
        raise ValueError(f"Extensión no soportada: {suffix} (usa .engine o .onnx)")

    def warmup(self, n: int = 5) -> None:
        import numpy as np

        kind, _ = self._require()
        if kind == "onnx":
            blank = np.zeros((1, 3, *self.input_size[::-1]), dtype="float32")
            for _ in range(n):
                self.infer(blank)

    def infer(self, frame: Any) -> list[dict]:
        kind, backend = self._require()
        if kind == "onnx":
            import numpy as np

            arr = frame if isinstance(frame, np.ndarray) else np.asarray(frame)
            out = backend.run(None, {backend.get_inputs()[0].name: arr})
            return [{"raw": o.shape} for o in out]
        raise NotImplementedError(
            "Inferencia TensorRT directa pendiente: genera el engine y "
            "conéctalo aquí vía trtexec/nvinfer o DeepStream."
        )

    def close(self) -> None:
        self._backend = None


class FakeDetector:
    """Detector sintético para probar el harness sin GPU ni pesos."""

    def __init__(self, name: str = "fake", latency_ms: float = 5.0) -> None:
        self.name = name
        self.latency_ms = latency_ms

    def warmup(self, n: int = 5) -> None:
        return None

    def infer(self, frame: Any) -> list[dict]:
        import time

        time.sleep(self.latency_ms / 1000.0)
        return [{"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.9}]

    def close(self) -> None:
        return None
