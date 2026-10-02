"""Simulador determinista de nodos de zona para pruebas sin hardware.

Modela RSSI con el modelo log-distancia: `rssi = tx - 10·n·log10(d) + ruido`.
No reemplaza una campaña de medición real: los valores de `tx_power_dbm`,
`path_loss_exp` y `noise_db` deben calibrarse con el hardware instalado (tag XIAO ESP32-C6).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from datetime import datetime, timedelta

from .models import Channel, TagObservation


class ZoneNodeSimulator:
    """Genera observaciones de nodos de zona para un tag en una posición dada."""

    def __init__(
        self,
        nodes: Mapping[str, tuple[float, float]],
        *,
        seed: int = 0,
        tx_power_dbm: float = -59.0,
        path_loss_exp: float = 2.0,
        noise_db: float = 0.0,
        sensitivity_dbm: float = -95.0,
        channel: Channel = Channel.BLE,
    ) -> None:
        self._nodes = dict(nodes)
        self._rng = random.Random(seed)
        self._tx = tx_power_dbm
        self._n = path_loss_exp
        self._noise = noise_db
        self._sensitivity = sensitivity_dbm
        self._channel = channel
        self._sequences: dict[str, int] = {}
        self._outages: set[str] = set()
        self._clock_offsets: dict[str, float] = {}

    def set_outage(self, node_id: str, down: bool) -> None:
        """Simula un nodo caído (no reporta nada)."""
        (self._outages.add if down else self._outages.discard)(node_id)

    def set_clock_offset(self, node_id: str, seconds: float) -> None:
        """Simula un reloj desviado en el nodo (timestamps adelantados/atrasados)."""
        self._clock_offsets[node_id] = seconds

    def next_sequence(self, tag_id: str) -> int:
        """Reserva el siguiente número de secuencia del tag (empieza en 1)."""
        seq = self._sequences.get(tag_id, 0) + 1
        self._sequences[tag_id] = seq
        return seq

    def advertise(
        self,
        tag_id: str,
        position: tuple[float, float],
        at: datetime,
        *,
        battery_pct: int | None = 90,
    ) -> list[TagObservation]:
        """Un anuncio del tag: lo oyen todos los nodos en rango, con la misma secuencia."""
        sequence = self.next_sequence(tag_id)
        heard: list[TagObservation] = []
        for node_id, (nx, ny) in self._nodes.items():
            if node_id in self._outages:
                continue
            distance = max(math.hypot(position[0] - nx, position[1] - ny), 0.1)
            rssi = self._tx - 10 * self._n * math.log10(distance)
            if self._noise:
                rssi += self._rng.gauss(0.0, self._noise)
            rssi_int = min(0, round(rssi))
            if rssi_int < self._sensitivity:
                continue
            offset = timedelta(seconds=self._clock_offsets.get(node_id, 0.0))
            heard.append(
                TagObservation(
                    tag_id=tag_id,
                    node_id=node_id,
                    rssi_dbm=rssi_int,
                    timestamp=at + offset,
                    sequence=sequence,
                    battery_pct=battery_pct,
                    channel=self._channel,
                )
            )
        return heard
