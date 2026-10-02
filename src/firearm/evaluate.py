"""Evaluación común para cualquier detector de armas (una etapa o dos etapas).

Mide la clase `weapon` sobre splits YOLO: AP50 (101 puntos), precisión y recall al
umbral, umbral calibrado en validación (F1 máximo), falsas alarmas en negativos y
frames detectados en un video. Así las variantes se comparan con el mismo código.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Det = tuple[float, list[float]]  # (conf, xyxy) de la clase weapon
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_split(
    root: str | Path, split: str, cls: int = 1
) -> list[tuple[Path, list[list[float]]]]:
    """Imágenes de `images/<split>` con sus cajas de la clase `cls` en píxeles."""
    import cv2

    root = Path(root)
    out = []
    for p in sorted(
        x for x in (root / "images" / split).iterdir() if x.suffix.lower() in IMAGE_EXTS
    ):
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Imagen ilegible: {p}")
        h, w = img.shape[:2]
        lab = root / "labels" / split / f"{p.stem}.txt"
        boxes = []
        for ln in lab.read_text().splitlines() if lab.exists() else []:
            if not ln.strip():
                continue
            c, cx, cy, bw, bh = (float(v) for v in ln.split())
            if int(c) == cls:
                boxes.append(
                    [
                        (cx - bw / 2) * w,
                        (cy - bh / 2) * h,
                        (cx + bw / 2) * w,
                        (cy + bh / 2) * h,
                    ]
                )
        out.append((p, boxes))
    return out


def predict(
    detect: Callable[[np.ndarray], list[dict]], items: Sequence[tuple[Path, Any]]
) -> list[list[Det]]:
    import cv2

    preds = []
    for p, _ in items:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Imagen ilegible: {p}")
        dets = detect(img)
        preds.append(
            [
                (float(d["conf"]), [float(v) for v in d["xyxy"]])
                for d in dets
                if d.get("label") == "weapon"
            ]
        )
    return preds


def _matches(
    preds: Sequence[Sequence[Det]],
    gts: Sequence[Sequence[list[float]]],
    iou: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    confs, tps = [], []
    n_gt = 0
    for p, g in zip(preds, gts, strict=True):
        n_gt += len(g)
        used = [False] * len(g)
        for c, box in sorted(p, key=lambda x: -x[0]):
            best, j = 0.0, -1
            for k, gb in enumerate(g):
                if not used[k]:
                    v = _iou(box, gb)
                    if v > best:
                        best, j = v, k
            hit = best >= iou
            if hit:
                used[j] = True
            confs.append(c)
            tps.append(hit)
    return np.asarray(confs), np.asarray(tps, dtype=bool), n_gt


def metrics(
    preds: Sequence[Sequence[Det]], gts: Sequence[Sequence[list[float]]], thr: float
) -> dict[str, float]:
    conf, tp, n_gt = _matches(preds, gts)
    order = np.argsort(-conf, kind="stable")
    ctp = np.cumsum(tp[order])
    cfp = np.cumsum(~tp[order])
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1)
    ap = 0.0
    if len(conf) and n_gt:
        env = np.maximum.accumulate(prec[::-1])[::-1]
        idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
        ap = float(np.mean([env[i] if i < len(env) else 0.0 for i in idx]))
    keep = conf >= thr
    t = int(tp[keep].sum())
    f = int(keep.sum()) - t
    return {
        "ap50": ap,
        "threshold": thr,
        "precision": t / (t + f) if t + f else 0.0,
        "recall": t / n_gt if n_gt else 0.0,
        "tp": t,
        "fp": f,
        "gt": n_gt,
    }


def calibrate(
    preds: Sequence[Sequence[Det]], gts: Sequence[Sequence[list[float]]]
) -> float:
    """Umbral de confianza con F1 máximo (en validación, nunca en test)."""
    best, best_f1 = 0.35, -1.0
    for thr in np.round(np.arange(0.1, 0.9, 0.025), 3):
        m = metrics(preds, gts, float(thr))
        p, r = m["precision"], m["recall"]
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1:
            best, best_f1 = float(thr), f1
    return best


def negatives_rate(preds: Sequence[Sequence[Det]], thr: float) -> dict[str, float]:
    hits = sum(1 for p in preds if any(c >= thr for c, _ in p))
    return {
        "images": len(preds),
        "with_false_alarm": hits,
        "rate": hits / len(preds) if preds else 0.0,
    }


def video_frames(
    detect: Callable[[np.ndarray], list[dict]], video: str | Path, thr: float
) -> list[int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    hits, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if any(d.get("label") == "weapon" and d["conf"] >= thr for d in detect(f)):
            hits.append(i)
        i += 1
    cap.release()
    return hits


def dump(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
