"""Vista dentro/fuera sobre la estimación de `LocationService`.

La localización (validación, suavizado, frescura, confianza) la hace `src/location`.
Esta vista solo añade, para un receptor único, la decisión dentro/fuera del área con
un umbral de entrada `enter_dbm` y otro de salida `enter_dbm - hysteresis_db`; entre
ambos se mantiene el estado anterior para que el RSSI no haga parpadear el resultado.
"""

from __future__ import annotations

from dataclasses import dataclass

from location.models import EstimateStatus, UnknownReason, ZoneEstimate


@dataclass(frozen=True)
class PresenceState:
    status: str  # "inside" | "outside" | "unknown"
    smoothed_dbm: float | None
    confidence: float  # confianza de la estimación de LocationService
    zone_id: str | None
    ts: float | None  # epoch s de la última evidencia
    reason: str | None = None  # UnknownReason de LocationService


class PresenceView:
    def __init__(self, enter_dbm: float, hysteresis_db: float) -> None:
        if hysteresis_db < 0:
            raise ValueError("hysteresis_db debe ser >= 0")
        self.enter_dbm = enter_dbm
        self.exit_dbm = enter_dbm - hysteresis_db
        self._inside: bool | None = None

    def update(self, estimate: ZoneEstimate) -> PresenceState:
        last = estimate.last_evidence_at
        ts = last.timestamp() if last is not None else None
        if estimate.status is EstimateStatus.UNKNOWN or not estimate.evidence:
            self._inside = None  # al volver se decide como la primera vez
            reason = estimate.unknown_reason
            status = "unknown" if reason is UnknownReason.TAG_MISSING else "outside"
            return PresenceState(
                status,
                None,
                estimate.confidence,
                None,
                ts,
                reason.value if reason else None,
            )
        rssi = max(e.smoothed_rssi_dbm for e in estimate.evidence)
        if self._inside is None:
            self._inside = rssi >= self.enter_dbm
        elif self._inside and rssi < self.exit_dbm:
            self._inside = False
        elif not self._inside and rssi >= self.enter_dbm:
            self._inside = True
        status = "inside" if self._inside else "outside"
        return PresenceState(status, rssi, estimate.confidence, estimate.zone_id, ts)
