"""Evaluación propia de detección: AP por clase, umbrales, falsos positivos y fallos.

Se usa la misma implementación para el modelo COCO de 80 clases y para el
afinado, porque ambos se proyectan al espacio de las 9 clases de `NAMES`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .classes import NAMES

Pred = tuple[int, float, list[float]]  # (cls, conf, xyxy)
Gt = tuple[int, list[float]]  # (cls, xyxy)
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class ClassMatches:
    """Para una clase: conf de cada predicción, si es TP a cada IoU, y nº de GT."""

    conf: np.ndarray
    tp: np.ndarray  # (n_pred, 10)
    n_gt: int


def match_class(
    preds: Mapping[str, Sequence[Pred]], gts: Mapping[str, Sequence[Gt]], cls: int
) -> ClassMatches:
    """Emparejamiento voraz por confianza, imagen a imagen, para cada umbral de IoU."""
    confs: list[np.ndarray] = []
    tps: list[np.ndarray] = []
    n_gt = 0
    for key in set(preds) | set(gts):
        p = sorted((x for x in preds.get(key, ()) if x[0] == cls), key=lambda x: -x[1])
        g = np.asarray(
            [x[1] for x in gts.get(key, ()) if x[0] == cls], dtype=float
        ).reshape(-1, 4)
        n_gt += len(g)
        if not p:
            continue
        ious = iou_matrix(np.asarray([x[2] for x in p], dtype=float), g)
        tp = np.zeros((len(p), len(IOU_THRESHOLDS)), dtype=bool)
        for k, thr in enumerate(IOU_THRESHOLDS):
            used = np.zeros(len(g), dtype=bool)
            for i in range(len(p)):
                if not len(g):
                    break
                cand = np.where(~used & (ious[i] >= thr))[0]
                if len(cand):
                    used[cand[np.argmax(ious[i, cand])]] = True
                    tp[i, k] = True
        tps.append(tp)
        confs.append(np.asarray([x[1] for x in p], dtype=float))
    if not tps:
        return ClassMatches(
            np.zeros(0), np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool), n_gt
        )
    return ClassMatches(np.concatenate(confs), np.concatenate(tps), n_gt)


def average_precision(m: ClassMatches) -> np.ndarray:
    """AP interpolado en 101 puntos (estilo COCO) para cada umbral de IoU."""
    if m.n_gt == 0 or len(m.conf) == 0:
        return np.zeros(len(IOU_THRESHOLDS))
    order = np.argsort(-m.conf, kind="stable")
    tp = m.tp[order].astype(float)
    ctp, cfp = np.cumsum(tp, 0), np.cumsum(1 - tp, 0)
    rec = ctp / m.n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    ap = np.zeros(len(IOU_THRESHOLDS))
    grid = np.linspace(0, 1, 101)
    for k in range(len(IOU_THRESHOLDS)):
        p = np.maximum.accumulate(prec[::-1, k])[::-1]
        idx = np.searchsorted(rec[:, k], grid, side="left")
        ap[k] = np.mean([p[i] if i < len(p) else 0.0 for i in idx])
    return ap


def pr_at(m: ClassMatches, thr: float) -> tuple[float, float, int, int]:
    """Precisión y recall a IoU 0,5 con umbral de confianza `thr`; también TP y FP."""
    keep = m.conf >= thr
    tp = int(m.tp[keep, 0].sum())
    fp = int(keep.sum()) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / m.n_gt if m.n_gt else 0.0
    return prec, rec, tp, fp


def best_f1_threshold(
    m: ClassMatches, grid: Sequence[float] = tuple(np.arange(0.05, 0.9, 0.025))
) -> float:
    best, best_f1 = 0.3, -1.0
    for thr in grid:
        p, r, _, _ = pr_at(m, float(thr))
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1:
            best, best_f1 = float(thr), f1
    return round(best, 3)


def evaluate(
    preds: Mapping[str, Sequence[Pred]],
    gts: Mapping[str, Sequence[Gt]],
    thresholds: Mapping[str, float],
    classes: Sequence[int] | None = None,
) -> dict:
    classes = list(range(len(NAMES))) if classes is None else list(classes)
    per: dict[str, dict] = {}
    for c in classes:
        m = match_class(preds, gts, c)
        if m.n_gt == 0:
            continue
        ap = average_precision(m)
        thr = thresholds.get(NAMES[c], 0.3)
        p, r, tp, fp = pr_at(m, thr)
        per[NAMES[c]] = {
            "gt": m.n_gt,
            "ap50": float(ap[0]),
            "ap50_95": float(ap.mean()),
            "threshold": thr,
            "precision": p,
            "recall": r,
            "tp": tp,
            "fp": fp,
        }
    return {
        "map50": float(np.mean([v["ap50"] for v in per.values()])) if per else 0.0,
        "map50_95": float(np.mean([v["ap50_95"] for v in per.values()]))
        if per
        else 0.0,
        "per_class": per,
    }


def negative_fp_rate(
    preds: Mapping[str, Sequence[Pred]], thresholds: Mapping[str, float]
) -> dict:
    """Imágenes sin ninguna clase objetivo en las que se reporta algo (por clase y total)."""
    any_hit = 0
    per: dict[str, int] = {}
    for dets in preds.values():
        hit = {NAMES[c] for c, s, _ in dets if s >= thresholds.get(NAMES[c], 0.3)}
        any_hit += bool(hit)
        for name in hit:
            per[name] = per.get(name, 0) + 1
    n = len(preds)
    return {
        "images": n,
        "images_with_fp": any_hit,
        "fp_image_rate": any_hit / n if n else 0.0,
        "per_class": per,
    }


VIS_BUCKETS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))
HEIGHT_BUCKETS = ((0, 50), (50, 100), (100, 200), (200, 10_000))


def person_recall_buckets(
    preds: Mapping[str, Sequence[Pred]], meta: Mapping[str, Sequence[dict]], thr: float
) -> dict:
    """Recall de `person` (IoU 0,5) por visibilidad y altura en píxeles (oclusión / lejanía)."""
    hits: list[tuple[float, float, bool]] = []
    for key, boxes in meta.items():
        p = sorted(
            (x for x in preds.get(key, ()) if x[0] == 0 and x[1] >= thr),
            key=lambda x: -x[1],
        )
        g = np.asarray([b["xyxy"] for b in boxes], dtype=float).reshape(-1, 4)
        ious = iou_matrix(np.asarray([x[2] for x in p], dtype=float).reshape(-1, 4), g)
        used = np.zeros(len(g), dtype=bool)
        for i in range(len(p)):
            cand = np.where(~used & (ious[i] >= 0.5))[0]
            if len(cand):
                used[cand[np.argmax(ious[i, cand])]] = True
        hits += [
            (b["visibility"], b["height"], bool(u))
            for b, u in zip(boxes, used, strict=True)
        ]

    def bucket(idx: int, ranges: Sequence[tuple[float, float]]) -> dict:
        out = {}
        for lo, hi in ranges:
            sel = [h for h in hits if lo <= h[idx] < hi]
            out[f"{lo}-{hi if hi < 10_000 else 'inf'}"] = {
                "gt": len(sel),
                "recall": sum(h[2] for h in sel) / len(sel) if sel else 0.0,
            }
        return out

    return {
        "visibility": bucket(0, VIS_BUCKETS),
        "height_px": bucket(1, HEIGHT_BUCKETS),
    }


def load_yolo_gt(
    labels_dir: str, images: Mapping[str, tuple[int, int]]
) -> dict[str, list[Gt]]:
    """Etiquetas YOLO → `(cls, xyxy)` en píxeles. `images`: stem → (ancho, alto)."""
    from pathlib import Path

    out: dict[str, list[Gt]] = {}
    for stem, (w, h) in images.items():
        f = Path(labels_dir) / f"{stem}.txt"
        rows = []
        for ln in f.read_text().splitlines() if f.exists() else []:
            if not ln.strip():
                continue
            c, cx, cy, bw, bh = ln.split()
            cxf, cyf, bwf, bhf = (
                float(cx) * w,
                float(cy) * h,
                float(bw) * w,
                float(bh) * h,
            )
            rows.append(
                (int(c), [cxf - bwf / 2, cyf - bhf / 2, cxf + bwf / 2, cyf + bhf / 2])
            )
        out[stem] = rows
    return out
