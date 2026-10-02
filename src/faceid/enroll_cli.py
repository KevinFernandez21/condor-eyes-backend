"""CLI de enrolamiento por webcam: `uv run python scripts/face_enroll.py --help`.

Captura N fotos de la webcam, detecta/alinea con YuNet, calcula embeddings, guarda solo
los vectores en el índice del embedder elegido y descarta las imágenes (memoria y disco:
nunca se escriben). Con `--dry-run` usa fotogramas sintéticos y un embedder falso: no
necesita cámara, modelos ni red, y no escribe nada.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from .embedders import EMBEDDER_NAMES, CloudConsentError, EmbedderEngine
from .enroll import EnrollError, enroll_person
from .vectorstore import MEMORY, FaceIndex, index_path

INDEX_DIR = "data/faceid"


def webcam_frames(
    camera: int, interval: float, timeout: float, warmup: float = 1.0
) -> Iterator[np.ndarray]:
    """Cede un fotograma cada `interval` s hasta `timeout`. Nada se guarda en disco."""
    import cv2  # importación diferida

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {camera}")
    try:
        start = time.monotonic()
        last = 0.0
        while time.monotonic() - start < timeout:
            ok, img = cap.read()
            now = time.monotonic()
            if not ok or now - start < warmup or now - last < interval:
                continue
            last = now
            print(f"  fotograma capturado ({now - start:.0f}s)", file=sys.stderr)
            yield img
    finally:
        cap.release()


class _SyntheticDetector:
    """Detector falso para `--dry-run`: toda imagen sintética contiene una cara."""

    class _Face:
        box = (10.0, 10.0, 80.0, 80.0)
        score = 0.95
        yaw_ratio = 0.0
        size = 80.0

    def detect(self, img: np.ndarray) -> tuple[np.ndarray, list[Any]]:
        return img, [self._Face()]


class _SyntheticEmbedder:
    cloud = False

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._rng = np.random.default_rng(0)

    def embed(self, img: np.ndarray, face: Any) -> np.ndarray:
        return self.embed_many([(img, face)])[0]

    def embed_many(self, items: Any) -> list[np.ndarray]:
        out = []
        for _ in items:
            v = np.ones(16) + self._rng.normal(size=16) * 0.05
            out.append((v / np.linalg.norm(v)).astype("float32"))
        return out


def _synthetic_frames(n: int) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(42)
    for _ in range(n):
        yield (rng.random((120, 120, 3)) * 255).astype("uint8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="face_enroll",
        description="Enrola a una persona con consentimiento: guarda solo vectores, nunca fotos",
    )
    p.add_argument("--person-id", required=True)
    p.add_argument(
        "--consent-ref", required=True, help="Referencia al consentimiento firmado"
    )
    p.add_argument("--embedder", default="sface", choices=EMBEDDER_NAMES)
    p.add_argument(
        "--cloud-consent",
        action="store_true",
        help="Consentimiento específico para enviar la cara a la nube (obligatorio con gemini)",
    )
    p.add_argument("--photos", type=int, default=5, help="Fotos a capturar")
    p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--retention-days", type=float, default=365)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval", type=float, default=1.5, help="Segundos entre fotos")
    p.add_argument(
        "--timeout", type=float, default=60.0, help="Tiempo máximo de captura"
    )
    p.add_argument("--index-dir", default=INDEX_DIR)
    p.add_argument("--actor", default="operador")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Sin cámara, modelos ni red; no escribe nada (índice en memoria)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.embedder == "gemini" and not a.cloud_consent:
        print(
            "Gemini envía la cara a la nube: añade --cloud-consent solo si la persona "
            "dio su consentimiento específico para ello (nivel gratuito: Google puede "
            "usar el contenido para mejorar sus productos).",
            file=sys.stderr,
        )
        return 2
    try:
        if a.dry_run:
            emb: Any = _SyntheticEmbedder(a.embedder)
            engine = EmbedderEngine(_SyntheticDetector(), emb)
            index = FaceIndex(MEMORY, emb.model_id)
            frames = _synthetic_frames(a.photos)
        else:
            from .embedders import build_embedder

            engine, emb = build_embedder(a.embedder, cloud_consent=a.cloud_consent)
            index = FaceIndex(
                index_path(a.index_dir, emb.model_id), emb.model_id, cloud=emb.cloud
            )
            print(
                f"Mira a la cámara; se capturarán {a.photos} fotos "
                f"(una cada {a.interval}s; gira ligeramente la cabeza entre fotos).",
                file=sys.stderr,
            )
            frames = webcam_frames(a.camera, a.interval, a.timeout)
        report = enroll_person(
            frames,
            engine,
            emb,
            index,
            a.person_id,
            a.consent_ref,
            n=a.photos,
            min_samples=a.min_samples,
            cloud_consent=a.cloud_consent,
            retention_days=a.retention_days,
            actor=a.actor,
        )
    except (EnrollError, CloudConsentError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(json.dumps({**report, "dry_run": a.dry_run}, ensure_ascii=False))
    return 0
