"""Índice vectorial local (numpy + SQLite) con búsqueda coseno y decisión de conjunto abierto.

Reglas heredadas de `EnrollmentStore` (issue #10):
- Solo se guardan **vectores**, la referencia de persona, la de consentimiento y la
  caducidad. Nunca imágenes.
- Enrolar exige referencia de consentimiento; `delete` y `purge_expired` borran de verdad.
- Cada operación se audita (JSONL) sin datos biométricos.

Un índice pertenece a **un solo embedder** (`model_id`): los vectores de modelos distintos
no son comparables. Un índice de nube (`cloud=True`) solo admite enrolar con
`cloud_consent=True`.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedders import CloudConsentError
from .verify import VerifyStatus, write_audit

MEMORY = ":memory:"


@dataclass(frozen=True, slots=True)
class Hit:
    person_id: str
    score: float


def index_path(root: str | Path, model_id: str) -> Path:
    """Un archivo por embedder: `index_<model_id saneado>.sqlite`."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
    return Path(root) / f"index_{safe}.sqlite"


class FaceIndex:
    def __init__(
        self,
        path: str | Path,
        model_id: str,
        cloud: bool = False,
        audit_path: str | Path | None = None,
    ) -> None:
        self.path = str(path)
        self.model_id = model_id
        self.cloud = cloud
        if self.path == MEMORY:
            self.audit_path: Path | None = Path(audit_path) if audit_path else None
        else:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.audit_path = (
                Path(audit_path) if audit_path else p.with_suffix(".audit.jsonl")
            )
        self._con = sqlite3.connect(self.path)
        if self.path != MEMORY:
            self._con.execute("PRAGMA journal_mode=WAL")
        self._con.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                vec BLOB NOT NULL,
                consent_ref TEXT NOT NULL,
                cloud_consent INTEGER NOT NULL,
                enrolled_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS vectors_person ON vectors(person_id);
            """
        )
        self._init_meta()

    def _meta(self, key: str) -> str | None:
        row = self._con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def _set_meta(self, key: str, value: str) -> None:
        self._con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
        self._con.commit()

    def _init_meta(self) -> None:
        stored = self._meta("model_id")
        if stored is None:
            self._set_meta("model_id", self.model_id)
            self._set_meta("cloud", "1" if self.cloud else "0")
        elif stored != self.model_id:
            raise ValueError(
                f"El índice es del modelo '{stored}', no de '{self.model_id}': "
                "los vectores de modelos distintos no son comparables"
            )

    @property
    def dim(self) -> int | None:
        v = self._meta("dim")
        return None if v is None else int(v)

    def _audit(self, action: str, **kw: object) -> None:
        if self.audit_path is not None:
            write_audit(self.audit_path, action, model_id=self.model_id, **kw)

    # -- escritura -------------------------------------------------------
    def add(
        self,
        person_id: str,
        vectors: Sequence[np.ndarray],
        consent_ref: str,
        retention_days: float = 365.0,
        cloud_consent: bool = False,
        now: float | None = None,
        actor: str = "system",
    ) -> None:
        """Enrola (o re-enrola, reemplazando) a una persona con sus vectores."""
        if not consent_ref.strip():
            raise ValueError("El enrolamiento exige una referencia de consentimiento")
        if not vectors:
            raise ValueError("Se necesita al menos un vector válido para enrolar")
        if self.cloud and not cloud_consent:
            raise CloudConsentError(
                "El índice es de un embedder en la nube: exige cloud_consent=True"
            )
        mat = np.asarray([np.asarray(v, dtype="float32").flatten() for v in vectors])
        if mat.ndim != 2 or not np.isfinite(mat).all():
            raise ValueError("Vectores inválidos o no finitos")
        if self.dim is not None and mat.shape[1] != self.dim:
            raise ValueError(
                f"Dimensión {mat.shape[1]} distinta a la del índice ({self.dim})"
            )
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        if (norms < 1e-9).any():
            raise ValueError("Vector nulo")
        mat = mat / norms
        now = time.time() if now is None else now
        expires = now + retention_days * 86400
        with self._con:
            self._con.execute("DELETE FROM vectors WHERE person_id=?", (person_id,))
            self._con.executemany(
                "INSERT INTO vectors (person_id, vec, consent_ref, cloud_consent,"
                " enrolled_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        person_id,
                        v.tobytes(),
                        consent_ref,
                        int(cloud_consent),
                        now,
                        expires,
                    )
                    for v in mat
                ],
            )
            self._con.execute(
                "INSERT OR REPLACE INTO meta VALUES ('dim', ?)", (str(mat.shape[1]),)
            )
        self._audit(
            "enroll",
            person_id=person_id,
            samples=len(mat),
            consent_ref=consent_ref,
            cloud_consent=cloud_consent,
            actor=actor,
        )

    def delete(self, person_id: str, actor: str = "system") -> bool:
        with self._con:
            n = self._con.execute(
                "DELETE FROM vectors WHERE person_id=?", (person_id,)
            ).rowcount
        self._audit("delete", person_id=person_id, existed=n > 0, actor=actor)
        return n > 0

    def purge_expired(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        gone = [
            r[0]
            for r in self._con.execute(
                "SELECT DISTINCT person_id FROM vectors WHERE expires_at <= ?", (now,)
            )
        ]
        if gone:
            with self._con:
                self._con.execute("DELETE FROM vectors WHERE expires_at <= ?", (now,))
            self._audit("purge_expired", person_ids=gone)
        return sorted(gone)

    # -- lectura ---------------------------------------------------------
    def count(self, now: float | None = None) -> int:
        """Número de personas vigentes (no caducadas)."""
        now = time.time() if now is None else now
        row = self._con.execute(
            "SELECT COUNT(DISTINCT person_id) FROM vectors WHERE expires_at > ?", (now,)
        ).fetchone()
        return int(row[0])

    def search(
        self, query: np.ndarray, k: int = 3, now: float | None = None
    ) -> list[Hit]:
        """Top-k **personas** por similitud coseno (mejor vector de cada una)."""
        now = time.time() if now is None else now
        rows = self._con.execute(
            "SELECT person_id, vec FROM vectors WHERE expires_at > ?", (now,)
        ).fetchall()
        if not rows:
            return []
        q = np.asarray(query, dtype="float32").flatten()
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        mat = np.vstack([np.frombuffer(r[1], dtype="float32") for r in rows])
        if mat.shape[1] != q.shape[0]:
            raise ValueError(
                "La dimensión de la consulta no coincide con la del índice"
            )
        sims = mat @ q
        best: dict[str, float] = {}
        for (pid, _), s in zip(rows, sims, strict=True):
            best[pid] = max(best.get(pid, -2.0), float(s))
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(p, s) for p, s in ranked]

    def close(self) -> None:
        self._con.close()


def decide_open_set(
    hits: Sequence[Hit], threshold: float, margin: float
) -> tuple[VerifyStatus, str | None, float, float]:
    """`(estado, persona, mejor_score, segundo_score)` en conjunto abierto.

    - `unknown`: nadie alcanza el umbral (o el índice está vacío).
    - `inconclusive`: el mejor supera el umbral pero el segundo está a menos de `margin`.
    - `match`: supera el umbral con margen suficiente.
    """
    if not hits:
        return VerifyStatus.UNKNOWN, None, 0.0, -1.0
    best = hits[0].score
    second = hits[1].score if len(hits) > 1 else -1.0
    if best < threshold:
        return VerifyStatus.UNKNOWN, None, best, second
    if best - second < margin:
        return VerifyStatus.INCONCLUSIVE, None, best, second
    return VerifyStatus.MATCH, hits[0].person_id, best, second
