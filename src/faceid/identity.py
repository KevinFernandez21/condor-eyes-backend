"""Rol `identity`: de un recorte de cara a metadata de identidad.

El recorte llega por el lado del **pipeline** (nunca por el bus). Este manejador calcula
el embedding, consulta el índice vectorial y devuelve un `MetadataEnvelope` con solo
metadata: estado, score, referencia de persona, modelo y `requires_operator=True`.
El payload es JSON estricto (primitivas finitas; fechas ISO-8601 como texto), porque el
bus valida el contenido. No hay ningún campo de permitir/denegar acceso.

La integración con el runtime de agentes queda como seguimiento: este módulo no toca
`bus/hub.py` ni `agents/`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from bus.hub import MetadataEnvelope

from .embedders import CloudConsentError, Embedder, EmbedderEngine
from .vectorstore import FaceIndex, decide_open_set
from .verify import VerificationResult, Verifier, VerifyPolicy, VerifyStatus


@dataclass
class IdentityPolicy:
    threshold: float = 0.65  # coseno; calibrar por embedder (no es transferible)
    margin: float = 0.03
    top_k: int = 3


def json_safe(obj: Any) -> Any:
    """Convierte a primitivas JSON finitas: numpy -> nativo, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


class IdentityHandler:
    def __init__(
        self,
        engine: EmbedderEngine,
        embedder: Embedder,
        index: FaceIndex,
        policy: IdentityPolicy | None = None,
        verify_policy: VerifyPolicy | None = None,
        allow_cloud: bool = False,
        source: str = "identity",
    ) -> None:
        if embedder.cloud and not allow_cloud:
            # Un probe en la nube enviaría la cara de cualquier transeúnte a un tercero.
            raise CloudConsentError(
                "Embedder en la nube: cada cara consultada saldría del dispositivo; "
                "exige allow_cloud=True explícito (solo para personas con consentimiento de nube)"
            )
        if index.model_id != embedder.model_id:
            raise ValueError(
                f"Índice de '{index.model_id}' incompatible con el embedder '{embedder.model_id}'"
            )
        self.embedder = embedder
        self.index = index
        self.policy = policy or IdentityPolicy()
        self.verifier = Verifier(engine, None, verify_policy)
        self.source = source

    def result(self, crop: Any, now: float | None = None) -> VerificationResult:
        p = self.policy
        status, emb, ev = self.verifier.embed_one(crop)
        ev = {**ev, "model_id": self.embedder.model_id}
        if status is not None or emb is None:
            return VerificationResult(
                status or VerifyStatus.INVALID_INPUT, threshold=p.threshold, evidence=ev
            )
        hits = self.index.search(emb, p.top_k, now)
        st, who, best, second = decide_open_set(hits, p.threshold, p.margin)
        ev = {**ev, "second_score": round(second, 4), "enrolled": self.index.count(now)}
        if st is VerifyStatus.INCONCLUSIVE:
            ev["reason"] = "ambiguous"
        return VerificationResult(st, who, best, p.threshold, ev)

    def identify(
        self,
        crop: Any,
        stream_id: str | None = None,
        now: float | None = None,
    ) -> MetadataEnvelope:
        now = time.time() if now is None else now
        res = self.result(crop, now)
        payload = json_safe(
            {
                **res.to_payload(),
                "model_id": self.embedder.model_id,
                "observed_at": datetime.fromtimestamp(now, UTC).isoformat(),
            }
        )
        return MetadataEnvelope(
            source=self.source, stream_id=stream_id, payload=payload
        )
