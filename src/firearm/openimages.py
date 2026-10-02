"""Open Images V7 → YOLO para el detector de armas (`0 = person`, `1 = weapon`).

- `weapon`: cajas de `Handgun`, `Rifle` y `Shotgun` que no son dibujos
  (`IsDepiction`) ni de grupo (`IsGroupOf`). Las imágenes cuyas armas son todas
  dibujos o grupos se descartan.
- Negativos difíciles: imágenes con teléfono, paraguas, herramientas, linterna,
  mando o cámara y **sin** ningún arma ni cuchillo; solo llevan sus personas.
- `person`: `Person`, `Man`, `Woman`, `Boy`, `Girl`, sin duplicados (IoU > 0,7).
- Se respetan los splits oficiales (train / validation / test) de Open Images.

Licencias: anotaciones CC BY 4.0; imágenes listadas por Google como CC BY 2.0.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

FIREARMS = {"/m/0gxl3": "Handgun", "/m/06c54": "Rifle", "/m/06nrc": "Shotgun"}
OTHER_WEAPONS = {"/m/083kb": "Weapon", "/m/04ctx": "Knife"}
PERSONS = {"/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"}
NEGATIVE_OBJECTS = {
    "/m/050k8": "Mobile phone",
    "/m/0hnnb": "Umbrella",
    "/m/07k1x": "Tool",
    "/m/03l9g": "Hammer",
    "/m/01kb5b": "Flashlight",
    "/m/0qjjc": "Remote control",
    "/m/0dv5r": "Camera",
    "/m/01j5ks": "Wrench",
    "/m/01bms0": "Screwdriver",
}
S3 = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

Box = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax normalizados


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _dedupe(boxes: Iterable[Box], thr: float = 0.7) -> list[Box]:
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(_iou(b, k) <= thr for k in kept):
            kept.append(b)
    return kept


def read_annotations(path: str | Path) -> dict[str, dict[str, Any]]:
    """Agrupa por imagen las filas del CSV filtrado de cajas de Open Images."""
    imgs: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "firearm": [],
            "firearm_dropped": 0,
            "person": [],
            "other_weapon": 0,
            "negative": set(),
        }
    )
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = r["LabelName"]
            box = (
                float(r["XMin"]),
                float(r["YMin"]),
                float(r["XMax"]),
                float(r["YMax"]),
            )
            im = imgs[r["ImageID"]]
            if label in FIREARMS:
                if r.get("IsDepiction") == "1" or r.get("IsGroupOf") == "1":
                    im["firearm_dropped"] += 1
                else:
                    im["firearm"].append(box)
            elif label in OTHER_WEAPONS:
                im["other_weapon"] += 1
            elif label in PERSONS:
                if r.get("IsGroupOf") != "1":
                    im["person"].append(box)
            elif label in NEGATIVE_OBJECTS:
                im["negative"].add(NEGATIVE_OBJECTS[label])
    return imgs


def select(
    imgs: dict[str, dict[str, Any]], max_weapon: int, max_negative: int
) -> tuple[list[str], list[str]]:
    """Orden determinista por id: primeras `max_weapon` con arma y `max_negative` negativas."""
    weapon = sorted(
        i for i, d in imgs.items() if d["firearm"] and not d["firearm_dropped"]
    )
    negative = sorted(
        i
        for i, d in imgs.items()
        if d["negative"]
        and not d["firearm"]
        and not d["firearm_dropped"]
        and not d["other_weapon"]
    )
    return weapon[:max_weapon], negative[:max_negative]


def _labels(d: dict[str, Any]) -> list[str]:
    lines = []
    for cls, boxes in ((0, _dedupe(d["person"])), (1, d["firearm"])):
        for x1, y1, x2, y2 in boxes:
            if x2 - x1 > 0 and y2 - y1 > 0:
                lines.append(
                    f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                )
    return lines


def _fetch(oi_split: str, image_id: str, dst: Path, max_side: int) -> bool:
    import cv2
    import numpy as np

    if dst.exists():
        return True
    try:
        with urllib.request.urlopen(
            S3.format(split=oi_split, image_id=image_id), timeout=60
        ) as r:
            data = r.read()
    except OSError:
        return False
    img = cv2.imdecode(np.frombuffer(data, dtype="uint8"), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:  # se reduce para ahorrar disco; las etiquetas son relativas y no cambian
        img = cv2.resize(
            img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.jpg")
    cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    tmp.replace(dst)
    return True


def build_split(
    ann_csv: str | Path,
    oi_split: str,
    out_dir: str | Path,
    split: str,
    max_weapon: int,
    max_negative: int,
    negatives_split: str | None = None,
    max_side: int = 960,
    threads: int = 24,
) -> dict[str, Any]:
    """Descarga las imágenes elegidas y escribe `images/<split>` y `labels/<split>`.

    Los negativos van a `negatives_split` si se indica (para medir falsos positivos
    aparte); si no, al mismo split, con etiqueta solo de personas.
    """
    out_dir = Path(out_dir)
    imgs = read_annotations(ann_csv)
    weapon, negative = select(imgs, max_weapon, max_negative)
    jobs = [(i, split) for i in weapon] + [
        (i, negatives_split or split) for i in negative
    ]

    def work(job: tuple[str, str]) -> tuple[str, str, bool]:
        image_id, target = job
        ok = _fetch(
            oi_split,
            image_id,
            out_dir / "images" / target / f"{image_id}.jpg",
            max_side,
        )
        if ok:
            lab = out_dir / "labels" / target / f"{image_id}.txt"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lines = _labels(imgs[image_id])
            lab.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return image_id, target, ok

    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(work, jobs))
    weapon_set = set(weapon)
    done = [(i, ok) for i, _, ok in results]
    return {
        "split": split,
        "oi_split": oi_split,
        "weapon_images": sum(1 for i, ok in done if ok and i in weapon_set),
        "negative_images": sum(1 for i, ok in done if ok and i not in weapon_set),
        "failed": sum(1 for _, ok in done if not ok),
        "weapon_boxes": sum(
            len(imgs[i]["firearm"]) for i, ok in done if ok and i in weapon_set
        ),
        "person_boxes": sum(len(_dedupe(imgs[i]["person"])) for i, ok in done if ok),
        "candidates_with_firearm": sum(1 for d in imgs.values() if d["firearm"]),
    }


def add_pseudo_persons(
    out_dir: str | Path,
    splits: Iterable[str],
    model: str = "weights/yolov8n.pt",
    conf: float = 0.5,
    iou_dup: float = 0.5,
    batch: int = 32,
) -> dict[str, int]:
    """Completa las personas sin etiquetar de Open Images con YOLOv8n COCO.

    Open Images no etiqueta a todas las personas de cada imagen; sin esto el modelo
    aprendería a ignorarlas. Solo se añaden detecciones `person` con `conf` alta que
    no solapen (IoU > `iou_dup`) con una persona ya anotada. Las armas no se tocan.
    """
    from ultralytics import YOLO

    yolo: Any = YOLO(model)
    out_dir = Path(out_dir)
    added: dict[str, int] = {}
    for split in splits:
        paths = sorted((out_dir / "images" / split).glob("*.jpg"))
        n = 0
        for i in range(0, len(paths), batch):
            chunk = paths[i : i + batch]
            for p, r in zip(
                chunk,
                yolo.predict(
                    [str(x) for x in chunk], classes=[0], conf=conf, verbose=False
                ),
                strict=True,
            ):
                lab = out_dir / "labels" / split / f"{p.stem}.txt"
                lines = [
                    ln
                    for ln in lab.read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
                people: list[Box] = []
                for ln in lines:
                    c, cx, cy, w, h = (float(v) for v in ln.split())
                    if c == 0:
                        people.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                for x1, y1, x2, y2 in r.boxes.xyxyn.tolist():
                    b = (x1, y1, x2, y2)
                    if all(_iou(b, q) <= iou_dup for q in people):
                        people.append(b)
                        lines.append(
                            f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
                        )
                        n += 1
                lab.write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
        added[split] = n
    return added


def write_combined_yaml(
    root: str | Path,
    name: str,
    splits: dict[str, list[str]],
    names: tuple[str, ...] = ("person", "weapon"),
) -> Path:
    """data YAML con varias carpetas por split, relativas a `root` (p. ej. `datasets/`)."""
    root = Path(root).resolve()
    lines = [f"path: {root.as_posix()}"]
    for key, dirs in splits.items():
        lines.append(f"{key}:")
        lines += [f"  - {d}" for d in dirs]
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    path = root / f"{name}.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summary(path: str | Path, report: list[dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
