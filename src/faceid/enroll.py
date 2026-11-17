"""Enrolamiento a partir de fotogramas: detecta, controla calidad, embebe y descarta la imagen.

Los fotogramas se procesan de uno en uno y no se guardan en ningún sitio: solo sobreviven
los vectores (y, hasta el lote de embedding, los recortes en memoria RAM).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .embedders import CloudConsentError, Embedder, EmbedderEngine
from .vectorstore import FaceIndex
from .verify import Verifier, VerifyPolicy


class EnrollError(RuntimeError):
    """No se consiguieron suficientes caras válidas para enrolar."""


def enroll_person(
    frames: Iterable[np.ndarray],
    engine: EmbedderEngine,
    embedder: Embedder,
    index: FaceIndex,
    person_id: str,
    consent_ref: str,
    *,
    n: int = 5,
    min_samples: int = 3,
    cloud_consent: bool = False,
    retention_days: float = 365.0,
    actor: str = "operador",
    policy: VerifyPolicy | None = None,
) -> dict[str, Any]:
    if not consent_ref.strip():
        raise ValueError("El enrolamiento exige una referencia de consentimiento")
    if embedder.cloud and not cloud_consent:
        # Antes de capturar o calcular nada: la cara no debe salir sin consentimiento de nube.
        raise CloudConsentError(
            "Embedder en la nube: exige consentimiento de nube explícito"
        )
    verifier = Verifier(engine, None, policy)
    kept: list[tuple[np.ndarray, Any]] = []
    rejected: dict[str, int] = {}
    for frame in frames:
        status, located, _ = verifier.locate_one(frame)
        if status is not None or located is None:
            key = status.value if status else "invalid_input"
            rejected[key] = rejected.get(key, 0) + 1
        else:
            kept.append(located)
        del frame  # el fotograma no se conserva
        if len(kept) >= n:
            break
    if len(kept) < min_samples:
        raise EnrollError(
            f"Solo {len(kept)} caras válidas (mínimo {min_samples}); rechazadas: {rejected}"
        )
    vectors = embedder.embed_many(kept)
    del kept  # descarta imágenes recortadas
    index.add(
        person_id,
        vectors,
        consent_ref,
        retention_days=retention_days,
        cloud_consent=cloud_consent,
        actor=actor,
    )
    return {
        "person_id": person_id,
        "model_id": embedder.model_id,
        "samples": len(vectors),
        "rejected": rejected,
    }
