"""Construcción de los splits YOLO: COCO (9 clases), MOT16 (persona, CCTV) y negativos.

Reglas de split (ver `docs/surveillance-detection.md`):

- COCO train2017 → train; COCO val2017 se parte por `image_id` par/impar en
  val (par) y test (impar), así que ninguna imagen de test se usa para ajustar.
- MOT16: las secuencias de cámara fija 02 y 09 son **held-out**; el resto va a train
  salvo el último 30 % de 04, que es val.
- Negativos: imágenes del test COCO sin ninguna de las 9 clases (para medir la tasa
  de falsos positivos).
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .classes import COCO_ID_TO_IDX, NAMES

# MOT: 1 peatón, 2 persona sobre vehículo, 7 persona estática.
MOT_PERSON_CLASSES = {1, 2, 7}
MOT_HELDOUT = ("MOT16-02", "MOT16-09")
MOT_VAL = "MOT16-04"


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _write_label(path: Path, lines: Iterable[str]) -> int:
    lines = list(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _xywh(b: list[float]) -> tuple[float, float, float, float]:
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def _yolo_line(
    idx: int, x: float, y: float, w: float, h: float, iw: float, ih: float
) -> str | None:
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
    return f"{idx} {cx:.6f} {cy:.6f} {(x2 - x1) / iw:.6f} {(y2 - y1) / ih:.6f}"


def coco_split(
    ann_json: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    split: str,
    image_filter: str = "all",
    max_images: int | None = None,
    negatives: int = 0,
    negatives_split: str | None = None,
    download: bool = False,
) -> dict[str, Any]:
    """Convierte COCO a YOLO con las 9 clases.

    `image_filter`: `all`, `even` o `odd` por `image_id`. Se toman las primeras
    `max_images` con alguna clase objetivo (orden por id); las imágenes sin ninguna
    se descartan salvo hasta `negatives`, que van con etiqueta vacía a
    `negatives_split` (o a `split`). Con `download` las imágenes que falten se bajan
    de `coco_url`.
    """
    coco = json.loads(Path(ann_json).read_text(encoding="utf-8"))
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    anns: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] in COCO_ID_TO_IDX and not a.get("iscrowd", 0):
            anns[a["image_id"]].append(a)
    crowd = {a["image_id"] for a in coco["annotations"] if a.get("iscrowd", 0)}
    imgs = sorted(coco["images"], key=lambda i: i["id"])
    if image_filter != "all":
        parity = 0 if image_filter == "even" else 1
        imgs = [i for i in imgs if i["id"] % 2 == parity]

    # 1) selección determinista; 2) descarga en paralelo de lo que falte; 3) escritura.
    chosen: list[dict] = []
    n_pos = n_neg = 0
    for img in imgs:
        has = img["id"] in anns
        if has and (max_images is None or n_pos < max_images):
            chosen.append(img)
            n_pos += 1
        elif not has and n_neg < negatives and img["id"] not in crowd:
            chosen.append(img)
            n_neg += 1
        if max_images is not None and n_pos >= max_images and n_neg >= negatives:
            break
    missing = [i for i in chosen if not (images_dir / i["file_name"]).exists()]
    if missing and download:
        from concurrent.futures import ThreadPoolExecutor

        images_dir.mkdir(parents=True, exist_ok=True)

        def fetch(i: dict) -> None:
            tmp = images_dir / (i["file_name"] + ".part")
            urllib.request.urlretrieve(i["coco_url"], tmp)
            tmp.replace(images_dir / i["file_name"])

        with ThreadPoolExecutor(16) as pool:
            list(pool.map(fetch, missing))

    per_class: Counter[str] = Counter()
    stats: dict[str, Any] = {
        "split": split,
        "images": 0,
        "boxes": 0,
        "negatives": 0,
        "missing": 0,
    }
    neg_split = negatives_split or split
    for img in chosen:
        has = img["id"] in anns
        src = images_dir / img["file_name"]
        if not src.exists():
            stats["missing"] += 1
            continue
        target = split if has else neg_split
        lines = []
        for a in anns.get(img["id"], []):
            idx = COCO_ID_TO_IDX[a["category_id"]]
            line = _yolo_line(idx, *_xywh(a["bbox"]), img["width"], img["height"])
            if line:
                lines.append(line)
                per_class[NAMES[idx]] += 1
        _place(src, out_dir / "images" / target / src.name)
        n = _write_label(out_dir / "labels" / target / f"{src.stem}.txt", lines)
        if has:
            stats["images"] += 1
            stats["boxes"] += n
        else:
            stats["negatives"] += 1
    stats["per_class"] = dict(per_class)
    return stats


def mot_split(
    seq_dirs: Iterable[str | Path],
    out_dir: str | Path,
    split: str,
    frame_step: int = 1,
    min_visibility: float = 0.0,
    frame_range: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """Frames de secuencias MOT16 (con `gt/gt.txt`) a YOLO, clase `person`.

    Guarda además `meta/<split>.json` con visibilidad y altura de cada caja para el
    análisis de fallos (oclusión, personas pequeñas o lejanas).
    """
    out_dir = Path(out_dir)
    stats: dict[str, Any] = {"split": split, "images": 0, "boxes": 0, "sequences": []}
    meta: dict[str, list[dict]] = {}
    for seq in map(Path, seq_dirs):
        info = dict(
            line.split("=", 1)
            for line in (seq / "seqinfo.ini").read_text().splitlines()
            if "=" in line
        )
        iw, ih, n = int(info["imWidth"]), int(info["imHeight"]), int(info["seqLength"])
        lo, hi = int(frame_range[0] * n) + 1, int(frame_range[1] * n)
        boxes: dict[int, list[tuple]] = defaultdict(list)
        for row in (seq / "gt" / "gt.txt").read_text().splitlines():
            fs, ts, xs, ys, ws, hs, consider, cls, vs = row.split(",")[:9]
            if (
                int(consider)
                and int(cls) in MOT_PERSON_CLASSES
                and float(vs) >= min_visibility
            ):
                boxes[int(fs)].append(
                    (float(xs), float(ys), float(ws), float(hs), float(vs), int(ts))
                )
        stats["sequences"].append(seq.name)
        for f in range(lo, hi + 1, frame_step):
            src = seq / "img1" / f"{f:06d}.jpg"
            if not src.exists():
                continue
            name = f"{seq.name}_{f:06d}"
            lines, m = [], []
            for x, y, w, h, vis, tid in boxes.get(f, []):
                line = _yolo_line(0, x, y, w, h, iw, ih)
                if line:
                    lines.append(line)
                    m.append(
                        {
                            "xyxy": [x, y, x + w, y + h],
                            "visibility": vis,
                            "height": h,
                            "track": tid,
                        }
                    )
            _place(src, out_dir / "images" / split / f"{name}.jpg")
            stats["boxes"] += _write_label(
                out_dir / "labels" / split / f"{name}.txt", lines
            )
            stats["images"] += 1
            meta[name] = m
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / f"{split}.json").write_text(json.dumps(meta), encoding="utf-8")
    return stats


def yolo_person_only(
    src: str | Path, out_dir: str | Path, split: str, person_idx: int = 0
) -> dict[str, Any]:
    """Copia un split YOLO dejando solo la clase persona (p. ej. Simuletic `0=person`)."""
    src, out_dir = Path(src), Path(out_dir)
    stats: dict[str, Any] = {"split": split, "images": 0, "boxes": 0}
    for img in sorted((src / "images").iterdir()):
        lab = src / "labels" / f"{img.stem}.txt"
        lines = [
            "0 " + ln.split(maxsplit=1)[1]
            for ln in (lab.read_text().splitlines() if lab.exists() else [])
            if ln.strip() and int(ln.split()[0]) == person_idx
        ]
        _place(img, out_dir / "images" / split / img.name)
        stats["boxes"] += _write_label(
            out_dir / "labels" / split / f"{img.stem}.txt", lines
        )
        stats["images"] += 1
    return stats


def write_yaml(out_dir: str | Path, name: str, splits: dict[str, list[str]]) -> Path:
    """data YAML de ultralytics; cada split puede combinar varias carpetas."""
    out_dir = Path(out_dir).resolve()
    lines = [f"path: {out_dir.as_posix()}"]
    for key, dirs in splits.items():
        lines.append(f"{key}:")
        lines += [f"  - images/{d}" for d in dirs]
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    path = out_dir / f"{name}.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
