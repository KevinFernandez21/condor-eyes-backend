"""Embedders intercambiables para identidad facial: SFace, CNN genérica y Gemini Embedding 2.

Todos cumplen el protocolo `Embedder` y devuelven vectores L2-normalizados. Vectores de
modelos distintos **no son comparables**, así que cada embedder tiene su `model_id` y su
propio índice (ver `vectorstore.FaceIndex`).

Gemini es **nube**: cada cara enviada sale del dispositivo. Por eso `GeminiEmbedder`
exige `cloud_consent=True` (consentimiento específico, distinto del local) y la clave
solo se lee de la variable de entorno `GEMINI_API_KEY`. En el nivel gratuito Google
puede usar el contenido enviado para mejorar sus productos.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

GEMINI_MODEL = (
    "gemini-embedding-2"  # GA abril 2026, ai.google.dev/gemini-api/docs/embeddings
)
GEMINI_DIM = 768  # 128-3072 admitido; 768 es una de las recomendadas por Google
GEMINI_USD_PER_IMAGE = 0.00012  # nivel de pago; ai.google.dev/gemini-api/docs/pricing
EMBEDDER_NAMES = ("sface", "sface_int8", "mobilenet_imagenet", "gemini")


class CloudConsentError(PermissionError):
    """Se intentó enviar o almacenar una cara de nube sin consentimiento de nube."""


class GeminiRateLimitError(RuntimeError):
    """Se agotaron los reintentos tras errores 429 del nivel gratuito."""


class Embedder(Protocol):
    model_id: str
    cloud: bool

    def embed(self, img: np.ndarray, face: Any) -> np.ndarray: ...

    def embed_many(
        self, items: Sequence[tuple[np.ndarray, Any]]
    ) -> list[np.ndarray]: ...


def _unit(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype="float32").flatten()
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError("Embedding nulo o no finito")
    return a / n


class LocalEmbedder:
    """Envuelve un `FaceEngine` local (SFace o CNN genérica)."""

    cloud = False

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.model_id: str = engine.embedder_name

    def embed(self, img: np.ndarray, face: Any) -> np.ndarray:
        return self.engine.embed(img, face)  # type: ignore[no-any-return]

    def embed_many(self, items: Sequence[tuple[np.ndarray, Any]]) -> list[np.ndarray]:
        return [self.embed(i, f) for i, f in items]


def crop_face_jpeg(
    img: np.ndarray,
    box: Sequence[float],
    margin: float = 0.25,
    size: int = 224,
    quality: int = 95,
) -> bytes:
    """Recorte de la cara (con margen) como JPEG en memoria: nada se escribe a disco."""
    import cv2

    h, w = img.shape[:2]
    x, y, bw, bh = box
    mx, my = bw * margin, bh * margin
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("Recorte de cara vacío")
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("No se pudo codificar el recorte")
    return bytes(buf.tobytes())


class GeminiEmbedder:
    """Gemini Embedding 2 vía `google-genai` (importado de forma diferida)."""

    cloud = True

    def __init__(
        self,
        *,
        cloud_consent: bool = False,
        model: str = GEMINI_MODEL,
        dim: int = GEMINI_DIM,
        batch_size: int = 8,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        min_interval: float = 0.0,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.cloud_consent = cloud_consent
        self.model = model
        self.dim = dim
        self.model_id = f"{model}@{dim}"
        self.batch_size = max(1, batch_size)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.min_interval = min_interval
        self._client = client
        self._sleep = sleep
        self._env = os.environ if env is None else env
        self._last_call = 0.0
        self.api_calls = 0
        self.images_sent = 0

    # -- clave y cliente -------------------------------------------------
    def _key(self) -> str:
        key = self._env.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY")
        return key

    def _redact(self, text: str) -> str:
        key = self._env.get("GEMINI_API_KEY", "")
        return text.replace(key, "***") if key else text

    def _get_client(self) -> Any:
        if self._client is None:
            key = self._key()
            from google import genai  # importación diferida: SDK opcional

            self._client = genai.Client(api_key=key)
        return self._client

    # -- API ------------------------------------------------------------
    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        text = str(exc)
        return (
            getattr(exc, "code", None) == 429
            or "RESOURCE_EXHAUSTED" in text
            or "429" in text
        )

    def _call(self, batch: list[bytes]) -> list[np.ndarray]:
        client = self._get_client()
        # Un Content por imagen: varias partes en un Content se agregan en UN solo vector.
        contents = [
            {"parts": [{"inline_data": {"data": b, "mime_type": "image/jpeg"}}]}
            for b in batch
        ]
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            try:
                self._last_call = time.monotonic()
                self.api_calls += 1
                resp = client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config={"output_dimensionality": self.dim},
                )
                break
            except Exception as exc:  # noqa: BLE001  SDK lanza tipos variados
                if not self._is_rate_limit(exc):
                    raise RuntimeError(
                        f"Error de Gemini ({type(exc).__name__}): {self._redact(str(exc))}"
                    ) from None
                if attempt == self.max_retries:
                    raise GeminiRateLimitError(
                        f"Límite de uso (429) tras {self.max_retries} reintentos"
                    ) from None
                self._sleep(min(self.base_delay * 2**attempt, self.max_delay))
        vecs = [e.values for e in (resp.embeddings or [])]
        if len(vecs) != len(batch):
            raise RuntimeError("Gemini devolvió un número inesperado de embeddings")
        self.images_sent += len(batch)
        return [_unit(v) for v in vecs]

    def embed_many(self, items: Sequence[tuple[np.ndarray, Any]]) -> list[np.ndarray]:
        if not self.cloud_consent:
            # Guardia dura: antes de codificar o enviar nada.
            raise CloudConsentError(
                "Gemini es una API en la nube: exige consentimiento de nube "
                "(cloud_consent=True), distinto del consentimiento local"
            )
        jpegs = [crop_face_jpeg(img, face.box) for img, face in items]
        out: list[np.ndarray] = []
        for i in range(0, len(jpegs), self.batch_size):
            out.extend(self._call(jpegs[i : i + self.batch_size]))
        return out

    def embed(self, img: np.ndarray, face: Any) -> np.ndarray:
        return self.embed_many([(img, face)])[0]


class EmbedderEngine:
    """Adapta un detector + un `Embedder` al protocolo `Engine` que usa `Verifier`."""

    def __init__(self, detector: Any, embedder: Embedder) -> None:
        self.detector = detector
        self.embedder = embedder

    def detect(self, img: np.ndarray) -> tuple[np.ndarray, list[Any]]:
        return self.detector.detect(img)  # type: ignore[no-any-return]

    def embed(self, img: np.ndarray, face: Any) -> np.ndarray:
        return self.embedder.embed(img, face)


def build_embedder(
    name: str,
    *,
    cloud_consent: bool = False,
    weights_root: str = "weights/faceid",
    gemini_dim: int = GEMINI_DIM,
) -> tuple[EmbedderEngine, Embedder]:
    """Construye (motor de detección + embedder, embedder) para un nombre de `EMBEDDER_NAMES`."""
    from .engine import FaceEngine

    if name == "gemini":
        gem = GeminiEmbedder(cloud_consent=cloud_consent, dim=gemini_dim)
        det = FaceEngine("detector_only", weights_root=weights_root)
        return EmbedderEngine(det, gem), gem
    if name not in EMBEDDER_NAMES:
        raise ValueError(f"Embedder desconocido: {name}")
    eng = FaceEngine(name, weights_root=weights_root)
    local = LocalEmbedder(eng)
    return EmbedderEngine(eng, local), local
