"""Calibración de umbral y FAR/FRR en identificación abierta 1:N (personal enrolado vs desconocidos).

- Sujetos de validación y de test **disjuntos** (sin fuga de identidad).
- En cada split, la mitad de los sujetos se enrola (5 imágenes) y la otra mitad
  son desconocidos que nunca se enrolan.
- FAR: sondas de desconocidos aceptadas como alguien + sondas de enrolados
  aceptadas como **otra** persona. FRR: sondas de enrolados que no terminan en
  `match` con su propia identidad (incluye inconcluso y sin cara).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .verify import VerifyPolicy, VerifyStatus

ENROLL_IMAGES = 5


@dataclass
class Probe:
    subject: int
    status: VerifyStatus | None  # None = embedding válido
    emb: np.ndarray | None
    quality: dict[str, Any]


def perturbations() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    import cv2

    lut = (np.linspace(0, 1, 256) ** 2.2 * 255 * 0.5).astype("uint8")

    def occlude(y0: float, y1: float) -> Callable[[np.ndarray], np.ndarray]:
        def f(img: np.ndarray) -> np.ndarray:
            out = img.copy()
            h = img.shape[0]
            out[int(y0 * h) : int(y1 * h)] = 0
            return out

        return f

    return {
        "original": lambda im: im,
        "blur": lambda im: cv2.GaussianBlur(im, (9, 9), 3.0),
        "dark": lambda im: cv2.LUT(im, lut),
        "occlusion_eyes": occlude(0.30, 0.50),  # gafas oscuras
        "occlusion_mouth": occlude(0.62, 1.0),  # mascarilla
    }


def load(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Imagen ilegible: {path}")
    return img


def embed_set(
    verifier: Any,
    root: Path,
    subjects: Sequence[int],
    images: Sequence[int],
    perturb: Callable[[np.ndarray], np.ndarray],
) -> list[Probe]:
    out = []
    for s in subjects:
        for i in images:
            st, emb, q = verifier.embed_one(perturb(load(root / str(s) / f"{i}.png")))
            out.append(Probe(s, st, emb, q))
    return out


def templates(enroll: Sequence[Probe]) -> tuple[list[int], np.ndarray]:
    by: dict[int, list[np.ndarray]] = {}
    for p in enroll:
        if p.emb is not None:
            by.setdefault(p.subject, []).append(p.emb)
    ids = sorted(by)
    mat = np.asarray([np.mean(by[s], axis=0) for s in ids], dtype="float32")
    return ids, mat / np.linalg.norm(mat, axis=1, keepdims=True)


def decide(
    probe: Probe, ids: Sequence[int], mat: np.ndarray, thr: float, margin: float
) -> tuple[str, int | None]:
    if probe.status is not None or probe.emb is None:
        return (probe.status.value if probe.status else "invalid_input"), None
    sims = mat @ probe.emb
    order = np.argsort(-sims)
    best = float(sims[order[0]])
    second = float(sims[order[1]]) if len(order) > 1 else -1.0
    if best < thr:
        return "unknown", None
    if best - second < margin:
        return "inconclusive", None
    return "match", ids[int(order[0])]


def rates(
    genuine: Sequence[Probe],
    unknown: Sequence[Probe],
    ids: Sequence[int],
    mat: np.ndarray,
    thr: float,
    margin: float,
) -> dict[str, Any]:
    g = [decide(p, ids, mat, thr, margin) for p in genuine]
    u = [decide(p, ids, mat, thr, margin) for p in unknown]
    false_acc = sum(
        d == "match" and who != p.subject
        for (d, who), p in zip(g, genuine, strict=True)
    )
    false_acc += sum(d == "match" for d, _ in u)
    true_acc = sum(
        d == "match" and who == p.subject
        for (d, who), p in zip(g, genuine, strict=True)
    )
    n = len(genuine) + len(unknown)

    def share(ds: list[tuple[str, int | None]], key: str) -> float:
        return sum(d == key for d, _ in ds) / len(ds) if ds else 0.0

    return {
        "threshold": thr,
        "margin": margin,
        "far": false_acc / n if n else 0.0,
        "frr": 1 - true_acc / len(genuine) if genuine else 0.0,
        "genuine_inconclusive": share(g, "inconclusive"),
        "genuine_no_face": share(g, "no_face"),
        "genuine_unknown": share(g, "unknown"),
        "unknown_correctly_rejected": share(u, "unknown"),
        "probes": {"genuine": len(genuine), "unknown": len(unknown)},
    }


def calibrate(
    genuine: Sequence[Probe],
    unknown: Sequence[Probe],
    ids: Sequence[int],
    mat: np.ndarray,
    target_far: float,
    margin: float,
) -> dict[str, Any]:
    """Umbral más bajo (menor FRR) cuyo FAR en validación no supera `target_far`."""
    best = None
    for thr in np.round(np.arange(0.15, 0.9, 0.01), 3):
        r = rates(genuine, unknown, ids, mat, float(thr), margin)
        if r["far"] <= target_far:
            best = r
            break
    return best or rates(genuine, unknown, ids, mat, 0.9, margin)


def split_subjects(subjects: Sequence[int]) -> tuple[list[int], list[int]]:
    """Mitad enrolada / mitad desconocida, intercaladas para no sesgar por id."""
    return list(subjects[0::2]), list(subjects[1::2])


def two_face_rate(verifier: Any, root: Path, subjects: Sequence[int]) -> dict[str, Any]:
    """Pega dos caras distintas en una imagen: el sistema debe decir `multiple_faces`."""
    counts: dict[str, int] = {}
    for a, b in zip(subjects[0::2], subjects[1::2], strict=False):
        img = np.hstack([load(root / str(a) / "6.png"), load(root / str(b) / "6.png")])
        st, _, _ = verifier.embed_one(img)
        key = st.value if st else "single_face"
        counts[key] = counts.get(key, 0) + 1
    n = sum(counts.values())
    return {
        "pairs": n,
        "outcomes": counts,
        "multiple_faces_rate": counts.get("multiple_faces", 0) / n if n else 0.0,
    }


def latency_ms(
    verifier: Any, root: Path, subjects: Sequence[int], n: int = 200
) -> dict[str, float]:
    imgs = [load(root / str(s) / "7.png") for s in list(subjects)[:20]]
    for im in imgs[:5]:
        verifier.embed_one(im)
    t = []
    for i in range(n):
        s = time.perf_counter()
        verifier.embed_one(imgs[i % len(imgs)])
        t.append((time.perf_counter() - s) * 1000)
    t.sort()
    return {"p50_ms": t[len(t) // 2], "p90_ms": t[int(0.9 * len(t))]}


def calibrate_sharpness(
    verifier: Any,
    policy: VerifyPolicy,
    root: Path,
    val_subjects: Sequence[int],
    probe_images: Sequence[int],
    percentile: float = 5.0,
) -> dict[str, Any]:
    """Fija `min_sharpness` al percentil `percentile` de las caras limpias de validación.

    La nitidez absoluta depende de la resolución y del reescalado, así que el umbral
    se calibra con datos (nunca con el test) en vez de fijarse a mano.
    """
    previous = policy.min_sharpness
    policy.min_sharpness = 0.0
    vals = [
        p.quality["sharpness"]
        for p in embed_set(
            verifier, root, val_subjects[::4], probe_images, lambda im: im
        )
        if p.quality and "sharpness" in p.quality
    ]
    policy.min_sharpness = float(np.percentile(vals, percentile)) if vals else previous
    return {
        "min_sharpness": round(policy.min_sharpness, 2),
        "percentile": percentile,
        "samples": len(vals),
    }


def run(
    verifier: Any,
    policy: VerifyPolicy,
    root: Path,
    val_subjects: Sequence[int],
    test_subjects: Sequence[int],
    probe_images: Sequence[int],
    target_far: float,
) -> dict[str, Any]:
    if set(val_subjects) & set(test_subjects):
        raise ValueError("fuga de identidad entre val y test")
    enroll_imgs = range(ENROLL_IMAGES)
    out_q = calibrate_sharpness(verifier, policy, root, val_subjects, probe_images)
    pert = perturbations()
    out: dict[str, Any] = {"quality_calibration": out_q}

    v_enr, v_unk = split_subjects(val_subjects)
    v_ids, v_mat = templates(
        embed_set(verifier, root, v_enr, enroll_imgs, pert["original"])
    )
    v_gen = embed_set(verifier, root, v_enr, probe_images, pert["original"])
    v_un = embed_set(verifier, root, v_unk, probe_images, pert["original"])
    cal = calibrate(v_gen, v_un, v_ids, v_mat, target_far, policy.margin)
    out["calibration_val"] = cal
    thr = cal["threshold"]

    t_enr, t_unk = split_subjects(test_subjects)
    t_ids, t_mat = templates(
        embed_set(verifier, root, t_enr, enroll_imgs, pert["original"])
    )
    out["enrolled_test"] = len(t_ids)
    out["test"] = {}
    for name, fn in pert.items():
        gen = embed_set(verifier, root, t_enr, probe_images, fn)
        unk = embed_set(verifier, root, t_unk, probe_images, fn)
        out["test"][name] = rates(gen, unk, t_ids, t_mat, thr, policy.margin)
        if name == "original":
            out["curve_test"] = [
                {
                    k: round(v, 4)
                    for k, v in rates(
                        gen, unk, t_ids, t_mat, float(x), policy.margin
                    ).items()
                    if k in ("threshold", "far", "frr")
                }
                for x in np.round(np.arange(0.2, 0.8, 0.05), 2)
            ]
            # Pose: se agrupan las sondas genuinas por giro estimado (sin perturbar).
            buckets: dict[str, list[Probe]] = {
                "frontal(<0.15)": [],
                "medio(0.15-0.35)": [],
                "perfil(>0.35)": [],
            }
            for p in gen:
                yaw = abs(p.quality.get("yaw", 0.0)) if p.quality else None
                if yaw is None:
                    continue
                key = (
                    "frontal(<0.15)"
                    if yaw < 0.15
                    else "medio(0.15-0.35)"
                    if yaw <= 0.35
                    else "perfil(>0.35)"
                )
                buckets[key].append(p)
            out["pose_test"] = {
                k: {
                    "probes": len(v),
                    **{
                        m: rates(v, [], t_ids, t_mat, thr, policy.margin)[m]
                        for m in ("frr", "genuine_inconclusive")
                    },
                }
                for k, v in buckets.items()
            }
    out["two_faces"] = two_face_rate(verifier, root, test_subjects)
    out["latency"] = latency_ms(verifier, root, test_subjects)
    return out
