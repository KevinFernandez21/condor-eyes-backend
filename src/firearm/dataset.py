"""Conversión de CCTV-Gun (COCO) al formato YOLO de ultralytics.

Estructura de salida:

    <out>/images/{train,val,test,negatives}/*.jpg|png
    <out>/labels/{train,val,test,negatives}/*.txt   (cls cx cy w h normalizados)
    <out>/data.yaml

`test` es el held-out entre dominios (UCF por defecto) y `negatives` contiene
imágenes sin arma (teléfonos, herramientas, manos vacías) con etiqueta vacía.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def coco_bbox_to_yolo(bbox: list[float], width: float, height: float) -> tuple[float, float, float, float] | None:
    """[x, y, w, h] en píxeles → (cx, cy, w, h) normalizados y recortados a la imagen."""
    x, y, w, h = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def category_map(coco: dict[str, Any]) -> tuple[dict[int, int], list[str]]:
    """Mapea los ids COCO (ordenados) a índices YOLO contiguos 0..n-1."""
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    return {c["id"]: i for i, c in enumerate(cats)}, [c["name"] for c in cats]


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def convert_split(
    coco_json: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    split: str,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Convierte un JSON COCO a `images/<split>` + `labels/<split>`.

    Si se pasa `names`, se verifica que las categorías coincidan para que todos
    los splits compartan índices. Las imágenes faltantes se cuentan, no abortan.
    """
    coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    cat_to_idx, cat_names = category_map(coco)
    if names is not None and names != cat_names:
        raise ValueError(f"Categorías distintas entre splits: {cat_names} vs {names}")

    images_dir, out_dir = Path(images_dir), Path(out_dir)
    anns: dict[Any, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        if not a.get("iscrowd", 0) and not a.get("ignore", 0):
            anns[a["image_id"]].append(a)

    counts = {"images": 0, "missing": 0, "boxes": 0, "skipped_boxes": 0}
    per_class = dict.fromkeys(cat_names, 0)
    for img in coco["images"]:
        src = images_dir / img["file_name"]
        if not src.exists():
            counts["missing"] += 1
            continue
        lines = []
        for a in anns.get(img["id"], []):
            box = coco_bbox_to_yolo(a["bbox"], img["width"], img["height"])
            if box is None:
                counts["skipped_boxes"] += 1
                continue
            idx = cat_to_idx[a["category_id"]]
            per_class[cat_names[idx]] += 1
            lines.append(f"{idx} " + " ".join(f"{v:.6f}" for v in box))
        _place(src, out_dir / "images" / split / src.name)
        label = out_dir / "labels" / split / f"{src.stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts["images"] += 1
        counts["boxes"] += len(lines)
    return {"split": split, **counts, "per_class": per_class, "names": cat_names}


def add_negatives(src_dir: str | Path, out_dir: str | Path, split: str = "negatives") -> dict[str, Any]:
    """Copia imágenes sin arma con etiqueta vacía (YOLO las trata como fondo)."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    n = 0
    for src in sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        _place(src, out_dir / "images" / split / src.name)
        label = out_dir / "labels" / split / f"{src.stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("", encoding="utf-8")
        n += 1
    return {"split": split, "images": n, "boxes": 0}


def write_data_yaml(out_dir: str | Path, names: list[str], splits: dict[str, str]) -> Path:
    """Escribe el data.yaml de ultralytics con ruta absoluta y los splits presentes."""
    out_dir = Path(out_dir).resolve()
    lines = [f"path: {out_dir.as_posix()}"]
    lines += [f"{key}: images/{split}" for key, split in splits.items()]
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    path = out_dir / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def prepare_cctv_gun(
    root: str | Path,
    out_dir: str | Path,
    pair: str = "mgd_usrt",
    heldout: str = "ucf",
    negatives: str | Path | None = None,
) -> dict[str, Any]:
    """Arma el dataset del prototipo desde el `data/` de CCTV-Gun.

    - train/val: `data/<pair>/annotations_{train,val}.json` (MGD + USRT).
    - test: `data/<heldout>/annotation_detection/annotations_all.json` (UCF completo,
      dominio no visto en entrenamiento).
    - negatives: carpeta opcional de imágenes sin arma.
    """
    root, out_dir = Path(root), Path(out_dir)
    images = root / "all_images"
    if not images.is_dir():
        raise FileNotFoundError(
            f"No existe {images}. Sigue dataset_instructions.md de CCTV-Gun para descargar MGD/USRT/UCF."
        )
    report: dict[str, Any] = {"root": str(root.resolve()), "pair": pair, "heldout": heldout, "splits": []}
    train = convert_split(root / pair / "annotations_train.json", images, out_dir, "train")
    names = train["names"]
    report["splits"].append(train)
    report["splits"].append(convert_split(root / pair / "annotations_val.json", images, out_dir, "val", names))
    heldout_json = root / heldout / "annotation_detection" / "annotations_all.json"
    report["splits"].append(convert_split(heldout_json, images, out_dir, "test", names))
    if negatives is not None:
        report["splits"].append(add_negatives(negatives, out_dir))
    write_data_yaml(out_dir, names, {"train": "train", "val": "val", "test": "test"})
    report["names"] = names
    (Path(out_dir) / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def scene_of(stem: str) -> str:
    """`Scene3_12` → `Scene3`: agrupa los frames de una misma escena."""
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def split_yolo_by_scene(
    src: str | Path,
    out_dir: str | Path,
    names: list[str],
    val_scenes: list[str],
    test_scenes: list[str],
) -> dict[str, Any]:
    """Reparte un dataset YOLO (`images/` + `labels/`) en train/val/test por escena.

    Separar por escena evita que frames casi idénticos caigan a la vez en train y
    en val/test e inflen las métricas.
    """
    src, out_dir = Path(src), Path(out_dir)
    overlap = set(val_scenes) & set(test_scenes)
    if overlap:
        raise ValueError(f"Escenas en val y test a la vez: {sorted(overlap)}")
    images = sorted(p for p in (src / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    found = {scene_of(p.stem) for p in images}
    unknown = (set(val_scenes) | set(test_scenes)) - found
    if unknown:
        raise ValueError(f"Escenas inexistentes: {sorted(unknown)}; hay {sorted(found)}")

    splits: dict[str, dict[str, Any]] = {
        s: {"split": s, "images": 0, "boxes": 0, "scenes": set(), "per_class": dict.fromkeys(names, 0)}
        for s in ("train", "val", "test")
    }
    for img in images:
        scene = scene_of(img.stem)
        split = "val" if scene in val_scenes else "test" if scene in test_scenes else "train"
        label_src = src / "labels" / f"{img.stem}.txt"
        text = label_src.read_text(encoding="utf-8") if label_src.exists() else ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            cid = int(ln.split()[0])
            if not 0 <= cid < len(names):
                raise ValueError(f"Clase {cid} fuera de rango en {label_src}")
            splits[split]["per_class"][names[cid]] += 1
        _place(img, out_dir / "images" / split / img.name)
        dst = out_dir / "labels" / split / f"{img.stem}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        st = splits[split]
        st["images"] += 1
        st["boxes"] += len(lines)
        st["scenes"].add(scene)
    for st in splits.values():
        st["scenes"] = sorted(st["scenes"])
    write_data_yaml(out_dir, names, {"train": "train", "val": "val", "test": "test"})
    report = {"src": str(src.resolve()), "names": names, "splits": list(splits.values())}
    (out_dir / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
