"""Benchmark de re-ID: recuperación (rank-1, mAP) y asociación con estado inconcluso."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .associate import (
    AssociatorConfig,
    CrossCameraAssociator,
    LinkStatus,
    TrackDescriptor,
)
from .market import Crop


def retrieval(
    qf: np.ndarray, q: Sequence[Crop], gf: np.ndarray, g: Sequence[Crop]
) -> dict[str, float]:
    """Protocolo Market-1501: se excluye la misma identidad en la misma cámara."""
    gp = np.asarray([c.pid for c in g])
    gc = np.asarray([c.cam for c in g])
    r1 = r5 = 0
    aps = []
    for i, c in enumerate(q):
        sim = gf @ qf[i]
        valid = ~((gp == c.pid) & (gc == c.cam))
        order = np.argsort(-sim[valid], kind="stable")
        match = (gp[valid] == c.pid)[order]
        if not match.any():
            continue
        hits = np.where(match)[0]
        r1 += hits[0] == 0
        r5 += hits[0] < 5
        prec = np.arange(1, len(hits) + 1) / (hits + 1)
        aps.append(prec.mean())
    n = len(aps)
    return {
        "queries": n,
        "rank1": r1 / n if n else 0.0,
        "rank5": r5 / n if n else 0.0,
        "map": float(np.mean(aps)) if aps else 0.0,
    }


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-9)


def tracks_from(
    feats: np.ndarray, crops: Sequence[Crop], t: float
) -> tuple[list[TrackDescriptor], dict[tuple[str, int], int]]:
    """Agrupa recortes por (identidad, cámara) como un track local por cámara.

    Los distractores (identidad 0) quedan como un track por imagen. Devuelve los
    descriptores y la identidad real de cada track (para medir, nunca para decidir).
    """
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, c in enumerate(crops):
        groups[(c.pid, c.cam, i if c.pid == 0 else -1)].append(i)
    out, truth = [], {}
    for n, ((pid, cam, _), idx) in enumerate(sorted(groups.items())):
        emb = _unit(feats[idx].mean(0, keepdims=True))[0]
        d = TrackDescriptor(f"cam{cam}", n, t, t, emb)
        out.append(d)
        truth[d.key] = pid
    return out, truth


def association(
    qf: np.ndarray,
    q: Sequence[Crop],
    gf: np.ndarray,
    g: Sequence[Crop],
    cfg: AssociatorConfig,
) -> dict[str, Any]:
    """Cada consulta se asocia contra la galería de las otras cámaras en dos escenarios:

    - `present`: su identidad está en la galería → enlace correcto, falso o perdido.
    - `absent`: se quitan sus tracks de la galería → cualquier enlace es falso.
    """
    gal, gtruth = tracks_from(gf, g, t=0.0)
    qry, qtruth = tracks_from(qf, q, t=1.0)
    qry = [
        TrackDescriptor(f"q-{d.stream_id}", 100_000 + d.track_id, 1.0, 1.0, d.embedding)
        for d in qry
    ]
    qtruth = {(f"q-{k[0]}", 100_000 + k[1]): v for k, v in qtruth.items()}
    assoc = CrossCameraAssociator(cfg)
    for d in gal:
        assoc.upsert(d)
    keys_by_cam: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for d in gal:
        keys_by_cam[d.stream_id].append(d.key)
    by_pid: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for k, pid in gtruth.items():
        by_pid[pid].append(k)
    c: dict[str, defaultdict[str, int]] = {
        "present": defaultdict(int),
        "absent": defaultdict(int),
    }
    for d in qry:
        pid = qtruth[d.key]
        if pid <= 0:
            continue
        qcam = d.stream_id[2:]
        same_cam = [k for k in by_pid[pid] if k[0] == qcam]
        other = [k for k in by_pid[pid] if k[0] != qcam]
        # La consulta no puede ver su propia cámara (otra vista del mismo track).
        excl_cam = keys_by_cam[qcam]
        if other:
            r = assoc.associate(d, exclude=excl_cam)
            c["present"][_outcome(r.status, r.candidate, gtruth, pid)] += 1
        r = assoc.associate(d, exclude=[*excl_cam, *other, *same_cam])
        c["absent"][
            "false_link" if r.status is LinkStatus.LINKED else "correct_reject"
        ] += 1
    pres, absn = c["present"], c["absent"]
    n_p, n_a = sum(pres.values()), sum(absn.values())
    return {
        "present": dict(pres),
        "absent": dict(absn),
        "correct_link_rate": pres["correct_link"] / n_p if n_p else 0.0,
        "false_link_rate_present": pres["false_link"] / n_p if n_p else 0.0,
        "missed_link_rate": pres["missed_link"] / n_p if n_p else 0.0,
        "false_link_rate_absent": absn["false_link"] / n_a if n_a else 0.0,
        "false_link_rate": (pres["false_link"] + absn["false_link"]) / (n_p + n_a)
        if n_p + n_a
        else 0.0,
    }


def _outcome(
    status: LinkStatus, cand: tuple[str, int] | None, truth: dict, pid: int
) -> str:
    if status is not LinkStatus.LINKED:
        return "missed_link"
    return (
        "correct_link" if cand is not None and truth.get(cand) == pid else "false_link"
    )


def calibrate(
    qf: np.ndarray,
    q: Sequence[Crop],
    gf: np.ndarray,
    g: Sequence[Crop],
    max_false_link: float = 0.01,
    thresholds: Sequence[float] = tuple(np.round(np.arange(0.3, 0.96, 0.025), 3)),
    margins: Sequence[float] = (0.0, 0.02, 0.05, 0.1),
) -> dict[str, Any]:
    """Umbral y margen que maximizan enlaces correctos con falsos enlaces ≤ `max_false_link`."""
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for thr in thresholds:
        for mg in margins:
            r = association(
                qf, q, gf, g, AssociatorConfig(threshold=float(thr), margin=mg)
            )
            cand = {"threshold": float(thr), "margin": mg, **r}
            if fallback is None or r["false_link_rate"] < fallback["false_link_rate"]:
                fallback = cand
            if r["false_link_rate"] <= max_false_link and (
                best is None or r["correct_link_rate"] > best["correct_link_rate"]
            ):
                best = cand
    chosen = best or fallback
    assert chosen is not None
    chosen["meets_target"] = best is not None
    return chosen


def latency(
    embed: Callable[[np.ndarray], np.ndarray], crops: np.ndarray, n: int = 200
) -> dict[str, float]:
    """Latencia por recorte (batch 1) y por lote de 16 recortes, con warmup."""
    import torch

    for _ in range(10):
        embed(crops[:1])
    one = []
    for i in range(n):
        s = time.perf_counter()
        embed(crops[i % len(crops) : i % len(crops) + 1])
        one.append((time.perf_counter() - s) * 1000)
    batch = []
    for i in range(0, min(len(crops), 16 * 30), 16):
        s = time.perf_counter()
        embed(crops[i : i + 16])
        batch.append((time.perf_counter() - s) * 1000)
    one.sort()
    batch.sort()
    out = {
        "batch1_p50_ms": one[len(one) // 2],
        "batch1_p90_ms": one[int(0.9 * len(one))],
        "batch16_p50_ms": batch[len(batch) // 2] if batch else 0.0,
    }
    if torch.cuda.is_available():
        out["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out
