"""Cola acotada que descarta lo más antiguo para no frenar al productor."""

from __future__ import annotations

import threading
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    """Cola FIFO de tamaño fijo; al llenarse descarta el elemento más viejo.

    El productor (lector de cámara) nunca se bloquea, de modo que una cámara
    lenta o un consumidor lento no detienen a las demás.
    """

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize debe ser >= 1")
        self._maxsize = maxsize
        self._items: deque[T] = deque()
        self._lock = threading.Lock()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def put(self, item: T) -> int:
        """Encola `item`; devuelve cuántos elementos viejos se descartaron."""
        with self._lock:
            dropped = 0
            while len(self._items) >= self._maxsize:
                self._items.popleft()
                dropped += 1
            self._items.append(item)
            return dropped

    def get_nowait(self) -> T | None:
        """Extrae el más antiguo o None si está vacía."""
        with self._lock:
            return self._items.popleft() if self._items else None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
