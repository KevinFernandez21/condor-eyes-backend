"""Enrolamiento con consentimiento y verificación 1:N contra el personal autorizado.

- Solo se guardan **embeddings** derivados (nunca la imagen), con referencia de
  consentimiento, fecha de alta y caducidad. Borrado y purga explícitos.
- Cada operación queda en un log de auditoría (JSONL) sin datos biométricos.
- La verificación devuelve **evidencia**, no decide acceso: `requires_operator`
  siempre es True y no hay ningún campo de permitir/denegar.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from bus.hub import MetadataEnvelope


class VerifyStatus(StrEnum):
    MATCH = "match"  # coincide con una persona enrolada
    UNKNOWN = "unknown"  # cara de calidad suficiente que no coincide con nadie
    INCONCLUSIVE = "inconclusive"  # calidad baja o dos candidatos demasiado parecidos
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    status: VerifyStatus
    person_id: str | None = None
    score: float = 0.0
    threshold: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "person_id": self.person_id,
            "score": round(float(self.score), 4),
            "threshold": float(self.threshold),
            "requires_operator": True,
            "evidence": self.evidence,
        }

    def to_envelope(
        self, stream_id: str | None = None, source: str = "identity"
    ) -> MetadataEnvelope:
        return MetadataEnvelope(
            source=source, stream_id=stream_id, payload=self.to_payload()
        )


class Engine(Protocol):
    def detect(self, img: np.ndarray) -> tuple[np.ndarray, list[Any]]: ...
    def embed(self, img: np.ndarray, face: Any) -> np.ndarray: ...


@dataclass
class VerifyPolicy:
    # Valores calibrados en validación con DigiFace (ver docs/reports/face-verification.md).
    threshold: float = 0.65  # coseno; FAR <= 0,1 % en val
    margin: float = 0.03  # distancia mínima al segundo enrolado
    min_face_px: float = 40.0
    min_det_score: float = 0.8
    min_sharpness: float = 4.4  # varianza del Laplaciano; percentil 5 de val
    max_yaw: float = 0.35  # giro aproximado (nariz vs ojos)


def sharpness(img: np.ndarray, box: Sequence[float]) -> float:
    import cv2

    x, y, w, h = (round(v) for v in box)
    crop = img[max(0, y) : y + h, max(0, x) : x + w]
    if crop.size == 0:
        return 0.0
    return float(
        cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    )


def restrict_permissions(path: str | Path) -> None:
    """Deja el archivo accesible solo al propietario (0600). Best effort.

    En Windows `chmod` solo gestiona el bit de solo lectura: ahí los permisos reales los
    dan las ACL de NTFS (restringir la carpeta de datos al responsable, p. ej. con `icacls`).
    """
    p = Path(path)
    if p.exists():
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass


def write_audit(path: str | Path, action: str, **kw: Any) -> None:
    """Añade una línea JSONL al log de auditoría. Nunca debe recibir datos biométricos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({"t": time.time(), "action": action, **kw}, ensure_ascii=False)
            + "\n"
        )
    restrict_permissions(path)


class EnrollmentStore:
    """Plantillas por persona en un archivo JSON local (fuera de Git) + auditoría."""

    def __init__(self, path: str | Path, audit_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.audit_path = (
            Path(audit_path) if audit_path else self.path.with_suffix(".audit.jsonl")
        )
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def _audit(self, action: str, **kw: Any) -> None:
        write_audit(self.audit_path, action, **kw)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")

    def enroll(
        self,
        person_id: str,
        embeddings: Sequence[np.ndarray],
        consent_ref: str,
        retention_days: float = 365.0,
        now: float | None = None,
        actor: str = "system",
    ) -> None:
        if not consent_ref.strip():
            raise ValueError("El enrolamiento exige una referencia de consentimiento")
        if not embeddings:
            raise ValueError("Se necesita al menos un embedding válido para enrolar")
        now = time.time() if now is None else now
        t = np.mean([e / np.linalg.norm(e) for e in embeddings], axis=0)
        self._data[person_id] = {
            "template": (t / np.linalg.norm(t)).round(6).tolist(),
            "samples": len(embeddings),
            "consent_ref": consent_ref,
            "enrolled_at": now,
            "expires_at": now + retention_days * 86400,
        }
        self._save()
        self._audit(
            "enroll",
            person_id=person_id,
            samples=len(embeddings),
            consent_ref=consent_ref,
            actor=actor,
        )

    def delete(self, person_id: str, actor: str = "system") -> bool:
        existed = self._data.pop(person_id, None) is not None
        self._save()
        self._audit("delete", person_id=person_id, existed=existed, actor=actor)
        return existed

    def purge_expired(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        gone = [p for p, d in self._data.items() if d["expires_at"] <= now]
        for p in gone:
            del self._data[p]
        if gone:
            self._save()
            self._audit("purge_expired", person_ids=gone)
        return gone

    def templates(self, now: float | None = None) -> tuple[list[str], np.ndarray]:
        now = time.time() if now is None else now
        ids = [p for p, d in self._data.items() if d["expires_at"] > now]
        mat = np.asarray([self._data[p]["template"] for p in ids], dtype="float32")
        return ids, mat

    def __contains__(self, person_id: str) -> bool:
        return person_id in self._data


class Verifier:
    def __init__(
        self,
        engine: Engine,
        store: EnrollmentStore | None = None,
        policy: VerifyPolicy | None = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.policy = policy or VerifyPolicy()

    def locate_one(
        self, img: Any
    ) -> tuple[VerifyStatus | None, tuple[np.ndarray, Any] | None, dict[str, Any]]:
        """Valida la entrada y devuelve `(imagen_escalada, cara)` de la única cara, o el motivo.

        Separa detección y control de calidad del embedding para poder agrupar
        embeddings en lote (p. ej. contra una API en la nube).
        """
        if (
            not isinstance(img, np.ndarray)
            or img.ndim != 3
            or img.shape[2] != 3
            or min(img.shape[:2]) < 16
        ):
            return (
                VerifyStatus.INVALID_INPUT,
                None,
                {"reason": "imagen BGR inválida o vacía"},
            )
        scaled, faces = self.engine.detect(img)
        if not faces:
            return VerifyStatus.NO_FACE, None, {}
        if len(faces) > 1:
            return VerifyStatus.MULTIPLE_FACES, None, {"faces": len(faces)}
        f = faces[0]
        p = self.policy
        q: dict[str, Any] = {
            "face_px": round(float(f.size), 1),
            "det_score": round(float(f.score), 3),
            "sharpness": round(sharpness(scaled, f.box), 1),
            "yaw": round(float(f.yaw_ratio), 3),
        }
        low = [
            name
            for name, bad in (
                ("small_face", f.size < p.min_face_px),
                ("low_detection_score", f.score < p.min_det_score),
                ("blurry", q["sharpness"] < p.min_sharpness),
                ("pose", abs(q["yaw"]) > p.max_yaw),
            )
            if bad
        ]
        q["quality_issues"] = low
        if low:
            return VerifyStatus.INCONCLUSIVE, None, q
        return None, (scaled, f), q

    def embed_one(
        self, img: Any
    ) -> tuple[VerifyStatus | None, np.ndarray | None, dict[str, Any]]:
        """Valida la entrada y devuelve el embedding de la única cara, o el motivo del fallo."""
        status, located, q = self.locate_one(img)
        if status is not None or located is None:
            return status, None, q
        return None, self.engine.embed(*located), q

    def verify(self, img: Any, now: float | None = None) -> VerificationResult:
        p = self.policy
        status, emb, ev = self.embed_one(img)
        if status is not None or emb is None:
            return VerificationResult(
                status or VerifyStatus.INVALID_INPUT, threshold=p.threshold, evidence=ev
            )
        if self.store is None:
            raise RuntimeError("verify() necesita un EnrollmentStore")
        ids, mat = self.store.templates(now)
        if not ids:
            return VerificationResult(
                VerifyStatus.UNKNOWN,
                threshold=p.threshold,
                evidence={**ev, "enrolled": 0},
            )
        sims = mat @ emb
        order = np.argsort(-sims)
        best = float(sims[order[0]])
        second = float(sims[order[1]]) if len(order) > 1 else -1.0
        ev = {**ev, "second_score": round(second, 4), "enrolled": len(ids)}
        if best < p.threshold:
            return VerificationResult(VerifyStatus.UNKNOWN, None, best, p.threshold, ev)
        if best - second < p.margin:
            return VerificationResult(
                VerifyStatus.INCONCLUSIVE,
                None,
                best,
                p.threshold,
                {**ev, "reason": "ambiguous"},
            )
        return VerificationResult(
            VerifyStatus.MATCH, ids[int(order[0])], best, p.threshold, ev
        )
