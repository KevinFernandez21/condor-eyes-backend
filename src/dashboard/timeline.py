"""Línea de tiempo por cámara: cubetas de un minuto y clase, acotada en memoria.

Cada cubeta guarda el **máximo simultáneo** visto de cada clase en ese minuto
(las detecciones llegan varias veces por segundo; sumarlas inflaría el dato).
La serie ``all`` suma las cámaras minuto a minuto.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from .cameras import CLASSES

BUCKET_S = 60


class Timeline:
    def __init__(self, retention_minutes: int = 180, max_cameras: int = 64) -> None:
        self._retention = retention_minutes
        self._max_cameras = max_cameras
        # cámara -> {minuto (índice) -> {clase -> máximo}}
        self._data: OrderedDict[str, dict[int, dict[str, int]]] = OrderedDict()
        self._latest = 0

    def add(self, camera_id: str, ts: float, counts: Mapping[str, int]) -> None:
        clean = {c: int(counts[c]) for c in CLASSES if int(counts.get(c, 0) or 0) > 0}
        if not clean:
            return
        series = self._data.get(camera_id)
        if series is None:
            if len(self._data) >= self._max_cameras:
                return
            series = self._data[camera_id] = {}
        minute = int(ts // BUCKET_S)
        self._latest = max(self._latest, minute)
        bucket = series.setdefault(minute, {})
        for cls, value in clean.items():
            bucket[cls] = max(bucket.get(cls, 0), value)
        floor = self._latest - self._retention
        for old in [m for m in series if m < floor]:
            del series[old]

    def size(self) -> int:
        return max((len(s) for s in self._data.values()), default=0)

    def snapshot(self, now: float, minutes: int = 60) -> dict[str, Any]:
        end = int(now // BUCKET_S)
        start = end - minutes + 1
        series: dict[str, list[dict[str, Any]]] = {}
        total: dict[int, dict[str, int]] = {}
        for camera_id, buckets in sorted(self._data.items()):
            rows = []
            for minute in sorted(m for m in buckets if start <= m <= end):
                counts = buckets[minute]
                rows.append(_row(minute, counts))
                acc = total.setdefault(minute, {})
                for cls, value in counts.items():
                    acc[cls] = acc.get(cls, 0) + value
            if rows:
                series[camera_id] = rows
        out = {"all": [_row(m, total[m]) for m in sorted(total)]}
        out.update(series)
        return {
            "from": _iso(start * BUCKET_S),
            "to": _iso((end + 1) * BUCKET_S),
            "bucket_s": BUCKET_S,
            "series": out,
        }


def _row(minute: int, counts: Mapping[str, int]) -> dict[str, Any]:
    return {"t": minute * BUCKET_S, **{c: counts.get(c, 0) for c in CLASSES}}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()
