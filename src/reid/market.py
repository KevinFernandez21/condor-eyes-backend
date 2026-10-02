"""Market-1501: lectura de nombres y splits por identidad sin solapamiento.

Nombre de archivo: `0002_c1s1_000451_03.jpg` → identidad 2, cámara 1. La identidad
`-1` es basura (se ignora) y `0000` son distractores (quedan en la galería).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

_NAME = re.compile(r"^(-?\d+)_c(\d)s(\d)_(\d+)_\d+")


@dataclass(frozen=True, slots=True)
class Crop:
    path: Path
    pid: int
    cam: int
    seq: int
    frame: int


def parse(path: Path) -> Crop | None:
    m = _NAME.match(path.name)
    if not m:
        return None
    pid, cam, seq, frame = (int(g) for g in m.groups())
    return Crop(path, pid, cam, seq, frame)


def load_dir(d: str | Path, keep_junk: bool = False) -> list[Crop]:
    crops = [c for p in sorted(Path(d).glob("*.jpg")) if (c := parse(p))]
    return crops if keep_junk else [c for c in crops if c.pid != -1]


@dataclass
class MarketSplits:
    train: list[Crop]
    val_query: list[Crop]
    val_gallery: list[Crop]
    test_query: list[Crop]
    test_gallery: list[Crop]

    def identities(self) -> dict[str, set[int]]:
        def ids(cs: list[Crop]) -> set[int]:
            return {c.pid for c in cs if c.pid > 0}

        return {
            "train": ids(self.train),
            "val": ids(self.val_query) | ids(self.val_gallery),
            "test": ids(self.test_query) | ids(self.test_gallery),
        }


def build_splits(root: str | Path, val_ids: int = 75, seed: int = 0) -> MarketSplits:
    """train/val salen de `bounding_box_train` separando identidades; test es el oficial."""
    root = Path(root)
    train_all = load_dir(root / "bounding_box_train")
    pids = sorted({c.pid for c in train_all})
    rng = random.Random(seed)
    val_set = set(rng.sample(pids, val_ids))
    train = [c for c in train_all if c.pid not in val_set]
    val = [c for c in train_all if c.pid in val_set]
    # En val: una imagen por (identidad, cámara) hace de consulta; el resto es galería.
    seen: set[tuple[int, int]] = set()
    vq: list[Crop] = []
    vg: list[Crop] = []
    for c in val:
        (vg if (c.pid, c.cam) in seen else vq).append(c)
        seen.add((c.pid, c.cam))
    return MarketSplits(
        train=train,
        val_query=vq,
        val_gallery=vg,
        test_query=load_dir(root / "query"),
        test_gallery=load_dir(root / "bounding_box_test"),
    )
