"""Entrada de datos reales: tracks en formato MOT y eventos etiquetados en CSV.

Permite correr el mismo benchmark sobre video real una vez etiquetado según
`docs/events-benchmark.md`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .model import EventType, GroundTruthEvent, TrackObservation


def load_mot_tracks(
    path: str | Path,
    stream_id: str,
    fps: float,
    width: int,
    height: int,
    authorized_ids: set[int] | None = None,
) -> list[TrackObservation]:
    """`frame,id,x,y,w,h,...` (MOTChallenge, frame base 1) → punto de apoyo normalizado."""
    authorized_ids = authorized_ids or set()
    out: list[TrackObservation] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            frame, tid = int(float(row[0])), int(float(row[1]))
            x, y, w, h = (float(v) for v in row[2:6])
            if tid < 0:
                continue
            out.append(
                TrackObservation(
                    stream_id,
                    tid,
                    (frame - 1) / fps,
                    (x + w / 2) / width,
                    (y + h) / height,
                    tid in authorized_ids,
                )
            )
    out.sort(key=lambda o: (o.t, o.track_id))
    return out


GT_COLUMNS = (
    "stream_id",
    "event_type",
    "zone",
    "start_s",
    "end_s",
    "track_ids",
    "authorized",
)


def load_ground_truth(path: str | Path) -> list[GroundTruthEvent]:
    """CSV con columnas `GT_COLUMNS`; `track_ids` separados por `;` (vacío para aglomeración)."""
    out: list[GroundTruthEvent] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(GT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
        for row in reader:
            ids = tuple(int(v) for v in row["track_ids"].split(";") if v.strip())
            out.append(
                GroundTruthEvent(
                    EventType(row["event_type"]),
                    row["stream_id"],
                    row["zone"],
                    float(row["start_s"]),
                    float(row["end_s"]),
                    ids,
                    row["authorized"].strip().lower()
                    in {"1", "true", "yes", "si", "sí"},
                )
            )
    return out
