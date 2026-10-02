"""Presencia dentro/fuera de un área a partir del RSSI, con histéresis.

Umbral de entrada `enter_dbm`; umbral de salida `enter_dbm - hysteresis_db`. Entre
ambos se mantiene el estado anterior, para que el RSSI (muy ruidoso) no haga
parpadear dentro/fuera. Es lógica provisional hasta integrar `src/location`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PresenceState:
    status: str  # "inside" | "outside" | "unknown"
    smoothed_dbm: float | None
    confidence: float
    ts: float  # epoch s de la última muestra (o del cálculo si no hay)
    reason: str | None = None  # "lost" si caducó la última señal


class PresenceTracker:
    def __init__(
        self,
        enter_dbm: float,
        hysteresis_db: float,
        window: int = 5,
        lost_after_s: float = 10.0,
        margin_full_db: float = 10.0,
    ) -> None:
        if hysteresis_db < 0:
            raise ValueError("hysteresis_db debe ser >= 0")
        if window < 1:
            raise ValueError("window debe ser >= 1")
        if lost_after_s <= 0 or margin_full_db <= 0:
            raise ValueError("lost_after_s y margin_full_db deben ser > 0")
        self.enter_dbm = enter_dbm
        self.exit_dbm = enter_dbm - hysteresis_db
        self._samples: deque[float] = deque(maxlen=window)
        self._lost_after_s = lost_after_s
        self._margin_full_db = margin_full_db
        self._inside: bool | None = None
        self._last_ts: float | None = None

    def update(self, rssi: float, ts: float) -> PresenceState:
        if self._last_ts is not None and ts - self._last_ts > self._lost_after_s:
            self._samples.clear()  # tras perder la señal no se mezcla con muestras viejas
            self._inside = None
        self._samples.append(float(rssi))
        self._last_ts = ts
        smoothed = sum(self._samples) / len(self._samples)
        if self._inside is None:
            self._inside = smoothed >= self.enter_dbm
        elif self._inside and smoothed < self.exit_dbm:
            self._inside = False
        elif not self._inside and smoothed >= self.enter_dbm:
            self._inside = True
        return self._build(smoothed, ts)

    def state(self, now: float) -> PresenceState:
        if self._last_ts is None:
            return PresenceState("unknown", None, 0.0, now)
        if now - self._last_ts > self._lost_after_s:
            return PresenceState("outside", None, 0.0, self._last_ts, reason="lost")
        smoothed = sum(self._samples) / len(self._samples)
        return self._build(smoothed, self._last_ts)

    def _build(self, smoothed: float, ts: float) -> PresenceState:
        margin = smoothed - self.enter_dbm if self._inside else self.exit_dbm - smoothed
        confidence = min(1.0, max(0.0, margin / self._margin_full_db))
        return PresenceState(
            "inside" if self._inside else "outside", smoothed, round(confidence, 3), ts
        )
