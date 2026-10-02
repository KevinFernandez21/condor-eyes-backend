"""Estimación de zona con suavizado, histéresis y confianza.

Algoritmo (por tag):

1. Por cada nodo se suaviza el RSSI de las muestras dentro de `window_s` con una
   media móvil exponencial (`alpha`). Un nodo es candidato si su última muestra
   tiene menos de `evidence_max_age_s`.
2. La puntuación de una zona es el mejor RSSI suavizado de sus nodos.
3. Gana la zona con mayor puntuación, salvo que la zona vigente siga siendo
   candidata y la ventaja sea menor que `hysteresis_db` (evita parpadeo).
4. `confidence = (0.6·fuerza + 0.4·margen) · frescura · muestras · penalización`.

Las estimaciones solo exponen el seudónimo del tag, nunca el ID en claro.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import LocationConfig
from .models import (
    EstimateStatus,
    Evidence,
    TagObservation,
    UnknownReason,
    ZoneEstimate,
)
from .privacy import Pseudonymizer

_MAX_SAMPLES_PER_NODE = 64
_STRENGTH_WEIGHT = 0.6
_MARGIN_WEIGHT = 0.4
_MIN_FRESHNESS = 0.5


@dataclass(slots=True)
class _TagState:
    samples: dict[str, list[tuple[datetime, int]]] = field(default_factory=dict)
    chosen_zone: str | None = None
    last_seen_zone: str | None = None
    last_seen_at: datetime | None = None
    battery_pct: int | None = None
    battery_at: datetime | None = None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ZoneEstimator:
    """Mantiene muestras por tag/nodo y calcula la zona más probable."""

    def __init__(self, config: LocationConfig, pseudonymizer: Pseudonymizer) -> None:
        self._cfg = config
        self._pseudo = pseudonymizer
        self._tags: dict[str, _TagState] = {}
        self._node_seen: dict[str, datetime] = {}

    # --- entrada ---------------------------------------------------------

    def touch_node(self, node_id: str, at: datetime) -> None:
        """Registra vida del nodo (heartbeat o cualquier paquete recibido)."""
        previous = self._node_seen.get(node_id)
        if previous is None or at > previous:
            self._node_seen[node_id] = at

    def add(self, obs: TagObservation, received_at: datetime | None = None) -> None:
        """Incorpora una observación ya validada."""
        self.touch_node(obs.node_id, received_at or obs.timestamp)
        state = self._tags.setdefault(obs.tag_id, _TagState())
        samples = state.samples.setdefault(obs.node_id, [])
        samples.append((obs.timestamp, obs.rssi_dbm))
        samples.sort(key=lambda sample: sample[0])
        del samples[:-_MAX_SAMPLES_PER_NODE]
        zone = self._cfg.zone_of_node(obs.node_id)
        if state.last_seen_at is None or obs.timestamp >= state.last_seen_at:
            state.last_seen_at = obs.timestamp
            state.last_seen_zone = zone
        if obs.battery_pct is not None and (
            state.battery_at is None or obs.timestamp >= state.battery_at
        ):
            state.battery_pct = obs.battery_pct
            state.battery_at = obs.timestamp

    def known_tags(self) -> set[str]:
        """IDs de tags con muestras en memoria."""
        return set(self._tags)

    def prune(self, now: datetime) -> None:
        """Elimina muestras y tags más viejos que `retention_s`."""
        limit = now - timedelta(seconds=self._cfg.retention_s)
        for tag_id in list(self._tags):
            state = self._tags[tag_id]
            for node_id in list(state.samples):
                kept = [s for s in state.samples[node_id] if s[0] >= limit]
                if kept:
                    state.samples[node_id] = kept
                else:
                    del state.samples[node_id]
            if not state.samples:
                del self._tags[tag_id]

    # --- salud de nodos --------------------------------------------------

    def node_online(self, node_id: str, now: datetime) -> bool:
        seen = self._node_seen.get(node_id)
        if seen is None:
            return False
        return (now - seen).total_seconds() <= self._cfg.node_timeout_s

    def _zone_offline(self, zone_id: str, now: datetime) -> bool:
        return not any(
            self.node_online(node, now) for node in self._cfg.nodes_of_zone(zone_id)
        )

    # --- salida ----------------------------------------------------------

    def estimate(self, tag_id: str, now: datetime) -> ZoneEstimate:
        """Calcula la estimación. Actualiza la zona vigente usada por la histéresis."""
        cfg = self._cfg
        tag_ref = self._pseudo.ref("tag", tag_id)
        state = self._tags.get(tag_id)
        if state is None:
            return self._unknown(tag_ref, now, UnknownReason.TAG_MISSING)

        evidence = self._collect_evidence(state, now)
        if not evidence:
            reason = UnknownReason.STALE_EVIDENCE
            if state.last_seen_zone is not None and self._zone_offline(
                state.last_seen_zone, now
            ):
                reason = UnknownReason.NODE_OUTAGE
            return self._unknown(tag_ref, now, reason, state)

        scores: dict[str, float] = {}
        for item in evidence:
            scores[item.zone_id] = max(
                scores.get(item.zone_id, float("-inf")), item.smoothed_rssi_dbm
            )
        best = max(scores, key=lambda zone: scores[zone])
        current = state.chosen_zone
        if (
            current is not None
            and current in scores
            and current != best
            and scores[best] - scores[current] < cfg.hysteresis_db
        ):
            best = current
        state.chosen_zone = best

        chosen = [e for e in evidence if e.zone_id == best]
        first = min(e.first_seen_at for e in chosen)
        last = max(e.last_seen_at for e in chosen)
        others = [score for zone, score in scores.items() if zone != best]

        strength = _clamp(
            (scores[best] - cfg.rssi_floor_dbm)
            / (cfg.rssi_strong_dbm - cfg.rssi_floor_dbm)
        )
        margin = (
            1.0
            if not others
            else _clamp((scores[best] - max(others)) / cfg.margin_full_db)
        )
        age = max(0.0, (now - last).total_seconds())
        freshness = 1.0 - (1.0 - _MIN_FRESHNESS) * _clamp(age / cfg.evidence_max_age_s)
        sample_factor = _clamp(sum(e.samples for e in chosen) / cfg.samples_full)
        degraded = any(
            self._zone_offline(neighbor, now) for neighbor in cfg.zones[best].neighbors
        )
        confidence = (
            (_STRENGTH_WEIGHT * strength + _MARGIN_WEIGHT * margin)
            * freshness
            * sample_factor
            * (cfg.outage_penalty if degraded else 1.0)
        )

        ordered = tuple(
            sorted(
                evidence,
                key=lambda e: (e.zone_id != best, -e.smoothed_rssi_dbm),
            )
        )
        return ZoneEstimate(
            tag_ref=tag_ref,
            status=EstimateStatus.LOCATED,
            zone_id=best,
            confidence=round(confidence, 4),
            computed_at=now,
            first_evidence_at=first,
            last_evidence_at=last,
            evidence=ordered,
            degraded=degraded,
            battery_pct=state.battery_pct,
            battery_low=state.battery_pct is not None
            and state.battery_pct <= cfg.battery_low_pct,
        )

    # --- internos --------------------------------------------------------

    def _collect_evidence(self, state: _TagState, now: datetime) -> list[Evidence]:
        cfg = self._cfg
        fresh_limit = now - timedelta(seconds=cfg.evidence_max_age_s)
        window_limit = now - timedelta(seconds=cfg.smoothing_window_s)
        evidence: list[Evidence] = []
        for node_id, samples in state.samples.items():
            zone = cfg.zone_of_node(node_id)
            if zone is None or not samples or samples[-1][0] < fresh_limit:
                continue
            window = [s for s in samples if s[0] >= window_limit] or [samples[-1]]
            smoothed = float(window[0][1])
            for _, rssi in window[1:]:
                smoothed = (
                    cfg.smoothing_alpha * rssi + (1 - cfg.smoothing_alpha) * smoothed
                )
            evidence.append(
                Evidence(
                    node_id=node_id,
                    zone_id=zone,
                    smoothed_rssi_dbm=round(smoothed, 2),
                    samples=len(window),
                    first_seen_at=window[0][0],
                    last_seen_at=window[-1][0],
                )
            )
        return evidence

    def _unknown(
        self,
        tag_ref: str,
        now: datetime,
        reason: UnknownReason,
        state: _TagState | None = None,
    ) -> ZoneEstimate:
        battery = state.battery_pct if state else None
        return ZoneEstimate(
            tag_ref=tag_ref,
            status=EstimateStatus.UNKNOWN,
            zone_id=None,
            confidence=0.0,
            computed_at=now,
            last_evidence_at=state.last_seen_at if state else None,
            unknown_reason=reason,
            battery_pct=battery,
            battery_low=battery is not None and battery <= self._cfg.battery_low_pct,
        )
