"""Validación de observaciones: stale, skew, duplicados, replay e imposibles.

La secuencia (`sequence`) la genera el tag y se incrementa en cada anuncio; el
mismo paquete puede ser oído legítimamente por varios nodos. Por eso el estado
anti-replay se lleva por par (tag, nodo) con aritmética de 32 bits circular.

Limitación conocida (el firmware del XIAO ESP32-C6 la evita con seq persistente en
NVS): un tag que se reinicia vuelve a `sequence` bajo y sus
paquetes se rechazan como replay hasta que el contador supere el último visto
o pase `retention_s`. Una autenticación criptográfica (HMAC por tag) queda como
mejora futura; ver `docs/location.md`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .config import LocationConfig
from .models import RejectReason, TagObservation

_SEQ_MOD = 2**32
_SEQ_HALF = 2**31


class ObservationValidator:
    """Valida observaciones y mantiene el estado mínimo para detectar anomalías."""

    def __init__(self, config: LocationConfig) -> None:
        self._cfg = config
        # (tag, nodo) -> (última secuencia aceptada, instante de recepción)
        self._seq: dict[tuple[str, str], tuple[int, datetime]] = {}
        # tag -> zona -> último timestamp aceptado en esa zona
        self._zone_ts: dict[str, dict[str, datetime]] = {}

    @property
    def tracked_pairs(self) -> int:
        """Pares (tag, nodo) con estado de secuencia en memoria."""
        return len(self._seq)

    def validate(
        self, obs: TagObservation, received_at: datetime
    ) -> RejectReason | None:
        """Devuelve el motivo de rechazo o None si es válida (y registra su estado)."""
        cfg = self._cfg
        zone = cfg.zone_of_node(obs.node_id)
        if zone is None:
            return RejectReason.UNKNOWN_NODE
        if not cfg.rssi_min_dbm <= obs.rssi_dbm <= cfg.rssi_max_dbm:
            return RejectReason.IMPOSSIBLE_VALUE
        if obs.battery_pct is not None and not 0 <= obs.battery_pct <= 100:
            return RejectReason.IMPOSSIBLE_VALUE

        age_s = (received_at - obs.timestamp).total_seconds()
        if age_s > cfg.max_age_s:
            return RejectReason.STALE
        if -age_s > cfg.future_tolerance_s:
            return RejectReason.CLOCK_SKEW

        key = (obs.tag_id, obs.node_id)
        previous = self._seq.get(key)
        if previous is not None:
            diff = (obs.sequence - previous[0]) % _SEQ_MOD
            if diff == 0:
                return RejectReason.DUPLICATE
            if diff >= _SEQ_HALF:
                return RejectReason.REPLAY

        presence = obs.rssi_dbm >= cfg.transition_min_rssi_dbm
        if presence:
            for other_zone, other_ts in self._zone_ts.get(obs.tag_id, {}).items():
                if other_zone == zone or cfg.are_adjacent(zone, other_zone):
                    continue
                gap = abs((obs.timestamp - other_ts).total_seconds())
                if gap < cfg.transition_min_s:
                    return RejectReason.IMPOSSIBLE_TRANSITION

        self._seq[key] = (obs.sequence, received_at)
        if presence:
            zones = self._zone_ts.setdefault(obs.tag_id, {})
            if zone not in zones or obs.timestamp > zones[zone]:
                zones[zone] = obs.timestamp
        return None

    def prune(self, now: datetime) -> None:
        """Descarta estado más viejo que `retention_s`."""
        limit = now - timedelta(seconds=self._cfg.retention_s)
        self._seq = {k: v for k, v in self._seq.items() if v[1] >= limit}
        for tag in list(self._zone_ts):
            kept = {z: ts for z, ts in self._zone_ts[tag].items() if ts >= limit}
            if kept:
                self._zone_ts[tag] = kept
            else:
                del self._zone_ts[tag]
