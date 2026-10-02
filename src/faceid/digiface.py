"""Descarga parcial de DigiFace-1M (caras 100 % sintéticas, Microsoft, no comercial).

El zip se lee por HTTP Range: solo se bajan las imágenes pedidas, no los 2,9 GB.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DIGIFACE_P1 = "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_0-1999_72_imgs.zip"


MARKER = "SYNTHETIC_DIGIFACE.json"


def is_synthetic_digiface(root: str | Path) -> bool:
    """True si `root` fue creado por `download_subset` desde DigiFace-1M (caras sintéticas)."""
    try:
        meta = json.loads((Path(root) / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and meta.get("dataset") == "digiface-1m"


class HttpRangeFile(io.RawIOBase):
    """Archivo remoto de solo lectura con `seek` por cabeceras HTTP Range."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}
        )
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def download_subset(
    out_dir: str | Path,
    subjects: range,
    per_subject: int,
    url: str = DIGIFACE_P1,
    threads: int = 32,
) -> int:
    """Baja `per_subject` imágenes de cada identidad de `subjects` a `out/<id>/<n>.png`."""
    out_dir = Path(out_dir)
    if url == DIGIFACE_P1:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / MARKER).write_text(
            json.dumps({"dataset": "digiface-1m", "synthetic": True, "url": url}),
            encoding="utf-8",
        )
    wanted = [f"{s}/{i}.png" for s in subjects for i in range(per_subject)]
    todo = [w for w in wanted if not (out_dir / w).exists()]
    if not todo:
        return 0
    chunks = [todo[i::threads] for i in range(threads)]

    def worker(names: list[str]) -> int:
        z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), 1 << 16))
        for n in names:
            dst = out_dir / n
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(z.read(n))
            tmp.replace(dst)
        return len(names)

    with ThreadPoolExecutor(threads) as pool:
        return sum(pool.map(worker, chunks))
