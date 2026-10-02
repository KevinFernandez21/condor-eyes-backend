"""Fuentes de anuncios: adaptador BLE del PC (bleak) y una falsa para pruebas."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Self

from .receiver import Advertisement, SourceClosed


class BleakAdvertisementSource:
    """Escanea con el adaptador Bluetooth del PC. `bleak` se importa en `__aenter__`."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Advertisement] = asyncio.Queue(maxsize=1000)
        self._scanner: Any = None

    async def __aenter__(self) -> Self:
        try:
            from bleak import BleakScanner
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError("Falta bleak: `uv add bleak`.") from exc

        def on_detect(device: Any, data: Any) -> None:
            adv = Advertisement(
                address=device.address,
                rssi=data.rssi,
                manufacturer_data=dict(data.manufacturer_data),
                received_at=time.time(),
            )
            try:
                self._queue.put_nowait(adv)
            except asyncio.QueueFull:  # pragma: no cover - el consumidor va atrasado
                pass

        self._scanner = BleakScanner(detection_callback=on_detect)
        await self._scanner.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._scanner is not None:
            await self._scanner.stop()

    async def next(self, timeout: float) -> Advertisement | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            return None


class FakeAdvertisementSource:
    """Reproduce una lista de anuncios; `None` simula un periodo sin anuncios.

    Si se pasa `clock` (dict con clave "t"), avanza como el reloj del receptor.
    """

    def __init__(
        self, items: list[Advertisement | None], clock: dict[str, float] | None = None
    ) -> None:
        self._items = deque(items)
        self._clock = clock

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def next(self, timeout: float) -> Advertisement | None:
        if not self._items:
            raise SourceClosed
        item = self._items.popleft()
        if self._clock is not None:
            if item is None:
                self._clock["t"] += timeout
            else:
                self._clock["t"] = max(self._clock["t"], item.received_at)
        return item
