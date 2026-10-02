"""Asociación entre cámaras: track local → candidato de otra cámara, o inconcluso.

Solo maneja metadata: descriptores numéricos (embeddings), IDs de track, tiempos y
zonas. Ningún recorte de imagen pasa por aquí ni por el bus.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from bus.hub import MetadataEnvelope


class LinkStatus(StrEnum):
    LINKED = "linked"
    # Hay candidatos pero la evidencia no alcanza (similitud baja o ambigüedad).
    INCONCLUSIVE = "inconclusive"
    # No queda ningún candidato compatible (galería vacía, viejo, otra zona...).
    NO_CANDIDATE = "no_candidate"


@dataclass(frozen=True, slots=True)
class TrackDescriptor:
    """Resumen de un track local publicado por el tracker de una cámara."""

    stream_id: str
    track_id: int
    t_first: float
    t_last: float
    embedding: np.ndarray
    zone: str | None = None

    def __post_init__(self) -> None:
        # El tracker real entrega np.int64/np.float32; el bus solo admite primitivas JSON.
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "t_first", float(self.t_first))
        object.__setattr__(self, "t_last", float(self.t_last))

    @property
    def key(self) -> tuple[str, int]:
        return (self.stream_id, self.track_id)


@dataclass(frozen=True, slots=True)
class Transition:
    """Paso permitido entre zonas de cámaras distintas y su tiempo de tránsito."""

    src: tuple[str, str | None]
    dst: tuple[str, str | None]
    min_s: float = 0.0
    max_s: float = 120.0


@dataclass(frozen=True, slots=True)
class AssociationResult:
    source: tuple[str, int]
    candidate: tuple[str, int] | None
    status: LinkStatus
    confidence: float
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_stream": self.source[0],
            "source_track": self.source[1],
            "candidate_stream": self.candidate[0] if self.candidate else None,
            "candidate_track": self.candidate[1] if self.candidate else None,
            "status": self.status.value,
            "confidence": round(float(self.confidence), 4),
            "evidence": dict(self.evidence),
        }

    def to_envelope(self, source: str = "tracker") -> MetadataEnvelope:
        return MetadataEnvelope(
            source=source, stream_id=self.source[0], payload=self.to_payload()
        )


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass
class AssociatorConfig:
    # Similitud coseno mínima para enlazar.
    threshold: float = 0.6
    # Diferencia mínima con el segundo mejor *track distinto* para no ser ambiguo.
    margin: float = 0.05
    # Un track sin actualizar en este tiempo sale de la galería (retención).
    max_age_s: float = 600.0
    # Si hay transiciones definidas, solo se consideran las permitidas.
    transitions: tuple[Transition, ...] = ()


class CrossCameraAssociator:
    def __init__(self, config: AssociatorConfig | None = None) -> None:
        self.config = config or AssociatorConfig()
        self._gallery: dict[tuple[str, int], TrackDescriptor] = {}

    def __len__(self) -> int:
        return len(self._gallery)

    def upsert(self, d: TrackDescriptor) -> None:
        """Alta o actualización de un track; el mismo (cámara, track) nunca se duplica."""
        prev = self._gallery.get(d.key)
        if prev is not None:
            emb = _unit(_unit(prev.embedding) + _unit(d.embedding))
            d = TrackDescriptor(
                d.stream_id,
                d.track_id,
                min(prev.t_first, d.t_first),
                max(prev.t_last, d.t_last),
                emb,
                d.zone or prev.zone,
            )
        self._gallery[d.key] = d

    def forget(self, key: tuple[str, int]) -> None:
        self._gallery.pop(key, None)

    def prune(self, now: float) -> int:
        """Elimina los descriptores viejos (política de retención). Devuelve cuántos."""
        stale = [
            k
            for k, d in self._gallery.items()
            if now - d.t_last > self.config.max_age_s
        ]
        for k in stale:
            del self._gallery[k]
        return len(stale)

    def _compatible(self, q: TrackDescriptor, g: TrackDescriptor) -> tuple[bool, str]:
        if g.stream_id == q.stream_id:
            return False, "same_camera"
        dt = q.t_first - g.t_last
        if dt < 0:
            return False, "overlaps_in_time"
        if q.t_first - g.t_last > self.config.max_age_s:
            return False, "stale"
        if not self.config.transitions:
            return True, ""
        for tr in self.config.transitions:
            if (
                tr.src[0] == g.stream_id
                and tr.dst[0] == q.stream_id
                and tr.src[1] in (None, g.zone)
                and tr.dst[1] in (None, q.zone)
            ):
                if tr.min_s <= dt <= tr.max_s:
                    return True, ""
                return False, "transit_time"
        return False, "no_transition"

    def associate(
        self, q: TrackDescriptor, exclude: Iterable[tuple[str, int]] = ()
    ) -> AssociationResult:
        excluded = set(exclude)
        rejected: dict[str, int] = {}
        cands: list[tuple[float, TrackDescriptor]] = []
        qe = _unit(q.embedding)
        for g in self._gallery.values():
            if g.key == q.key or g.key in excluded:
                continue
            ok, why = self._compatible(q, g)
            if not ok:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            cands.append((float(qe @ _unit(g.embedding)), g))
        evidence: dict[str, Any] = {"candidates": len(cands), "rejected": rejected}
        if not cands:
            return AssociationResult(
                q.key, None, LinkStatus.NO_CANDIDATE, 0.0, evidence
            )
        cands.sort(key=lambda c: -c[0])
        best_sim, best = cands[0]
        second = cands[1][0] if len(cands) > 1 else -1.0
        evidence.update(
            {
                "similarity": round(best_sim, 4),
                "second_similarity": round(second, 4),
                "margin": round(best_sim - second, 4),
                "dt_s": round(q.t_first - best.t_last, 3),
                "candidate_zone": best.zone,
            }
        )
        cfg = self.config
        if best_sim < cfg.threshold:
            evidence["reason"] = "low_similarity"
            return AssociationResult(
                q.key, best.key, LinkStatus.INCONCLUSIVE, best_sim, evidence
            )
        if best_sim - second < cfg.margin:
            evidence["reason"] = "ambiguous"
            return AssociationResult(
                q.key, best.key, LinkStatus.INCONCLUSIVE, best_sim, evidence
            )
        conf = min(
            1.0, max(0.0, (best_sim - cfg.threshold) / max(1e-6, 1 - cfg.threshold))
        )
        return AssociationResult(
            q.key, best.key, LinkStatus.LINKED, 0.5 + 0.5 * conf, evidence
        )
