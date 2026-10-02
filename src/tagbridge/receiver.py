"""Receptor fijo: filtra el tag enrolado y lo convierte en observaciones payload v1.

El contrato JSON v1 es el de `docs/location.md` (issue #16). Los dispositivos BLE
que no son el tag enrolado se descartan sin registrar nada (ni log ni consola):
solo se cuenta cuántos se ignoraron.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, Self

from .presence import PresenceState, PresenceTracker
from .protocol import COMPANY_ID, SEQ_MASK, decode_tag_payload, normalize_tag

_NODE_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
# Un seq hasta 8 por detrás del último se considera repetido/desordenado; más atrás
# se asume reinicio del tag (el contador vuelve a empezar) y se acepta.
REORDER_WINDOW = 8


@dataclass(frozen=True)
class Advertisement:
    address: str
    rssi: int
    manufacturer_data: dict[int, bytes]
    received_at: float  # epoch s del reloj del receptor


@dataclass(frozen=True)
class ReceiverEvent:
    observation: dict[str, Any] | None  # None en ticks sin anuncio nuevo
    state: PresenceState


class SourceClosed(Exception):
    """La fuente de anuncios terminó."""


class AdvertisementSource(Protocol):
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *exc: object) -> None: ...
    async def next(self, timeout: float) -> Advertisement | None:
        """Siguiente anuncio, `None` si pasó `timeout`; `SourceClosed` al terminar."""
        ...


class TagReceiver:
    def __init__(
        self,
        enrolled: Iterable[str],
        node_id: str,
        tracker: PresenceTracker,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not _NODE_RE.match(node_id):
            raise ValueError(f"node_id inválido: {node_id!r}")
        self._enrolled = {normalize_tag(t) for t in enrolled}
        self._node_id = node_id
        self._tracker = tracker
        self._clock = clock
        self._last_seq: dict[str, int] = {}
        self.stats = {"accepted": 0, "ignored": 0, "duplicates": 0}

    def on_advertisement(self, adv: Advertisement) -> ReceiverEvent | None:
        raw = adv.manufacturer_data.get(COMPANY_ID)
        payload = decode_tag_payload(raw) if raw is not None else None
        if payload is None or payload.tag not in self._enrolled:
            self.stats["ignored"] += 1
            return None
        last = self._last_seq.get(payload.tag)
        if last is not None and ((last - payload.seq) & SEQ_MASK) <= REORDER_WINDOW:
            self.stats["duplicates"] += 1
            return None
        self._last_seq[payload.tag] = payload.seq
        self.stats["accepted"] += 1
        observation: dict[str, Any] = {
            "v": 1,
            "ch": "ble",
            "tag": payload.tag,
            "node": self._node_id,
            "rssi": adv.rssi,
            "ts": round(adv.received_at * 1000),
            "seq": payload.seq,
        }
        if payload.battery_pct is not None:
            observation["bat"] = payload.battery_pct
        state = self._tracker.update(adv.rssi, adv.received_at)
        return ReceiverEvent(observation, state)

    def tick(self) -> ReceiverEvent:
        return ReceiverEvent(None, self._tracker.state(self._clock()))


async def run_receiver(
    source: AdvertisementSource,
    receiver: TagReceiver,
    sink: Callable[[ReceiverEvent], None],
    tick_s: float = 1.0,
    duration_s: float | None = None,
) -> None:
    """Consume la fuente hasta que cierre o pasen `duration_s` segundos."""
    start = time.monotonic()
    async with source:
        while duration_s is None or time.monotonic() - start < duration_s:
            try:
                adv = await source.next(tick_s)
            except SourceClosed:
                return
            event = receiver.on_advertisement(adv) if adv is not None else None
            if adv is None:
                event = receiver.tick()
            if event is not None:
                sink(event)
