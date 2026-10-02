"""Benchmark comparativo de embedders (SFace vs Gemini Embedding 2 vs CNN genérica).

Reutiliza `benchmark.py` (mismos Probe/templates/calibrate/rates y el mismo reparto de
sujetos disjuntos val/test) y mide además latencia y coste estimado. Si un embedder no se
puede ejecutar (p. ej. Gemini sin `GEMINI_API_KEY`) se informa `"no ejecutado"`: nunca se
inventan números.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .benchmark import Probe, calibrate, load, rates, split_subjects, templates
from .embedders import GEMINI_USD_PER_IMAGE, Embedder, EmbedderEngine, build_embedder
from .verify import Verifier, VerifyPolicy

SINGLE_LATENCY_SAMPLES = 20


def skipped_report(model_id: str, reason: str) -> dict[str, Any]:
    return {"status": "no ejecutado", "model_id": model_id, "reason": reason}


def _percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def run_embedder(
    engine: EmbedderEngine,
    embedder: Embedder,
    root: Path,
    *,
    val_subjects: Sequence[int],
    test_subjects: Sequence[int],
    enroll_images: Sequence[int],
    probe_images: Sequence[int],
    target_far: float,
    margin: float = 0.03,
    price_per_image_usd: float = 0.0,
    policy: VerifyPolicy | None = None,
) -> dict[str, Any]:
    assert not set(val_subjects) & set(test_subjects), (
        "fuga de identidad entre val y test"
    )
    policy = policy or VerifyPolicy(margin=margin)
    verifier = Verifier(engine, None, policy)

    # 1) Detección + calidad (local, idéntica para todos los embedders).
    v_enr, v_unk = split_subjects(val_subjects)
    t_enr, t_unk = split_subjects(test_subjects)
    wanted: list[tuple[int, int]] = [
        (s, i) for s in (*v_enr, *t_enr) for i in (*enroll_images, *probe_images)
    ] + [(s, i) for s in (*v_unk, *t_unk) for i in probe_images]
    located: dict[tuple[int, int], tuple[Any, Any]] = {}
    failed: dict[tuple[int, int], tuple[Any, dict[str, Any]]] = {}
    for s, i in wanted:
        st, loc, q = verifier.locate_one(load(root / str(s) / f"{i}.png"))
        if st is not None or loc is None:
            failed[(s, i)] = (st, q)
        else:
            located[(s, i)] = loc

    # 2) Embeddings en lote (mismo camino que usaría la nube: batching + backoff).
    keys = list(located)
    t0 = time.perf_counter()
    vecs = embedder.embed_many([located[k] for k in keys]) if keys else []
    batch_s = time.perf_counter() - t0
    emb = dict(zip(keys, vecs, strict=True))

    # 3) Latencia de llamada individual (incluye red en la nube).
    singles: list[float] = []
    for k in keys[:SINGLE_LATENCY_SAMPLES]:
        t = time.perf_counter()
        embedder.embed(*located[k])
        singles.append((time.perf_counter() - t) * 1000)
    images_embedded = len(vecs) + len(singles)

    def probes(subjects: Sequence[int], images: Sequence[int]) -> list[Probe]:
        out = []
        for s in subjects:
            for i in images:
                if (s, i) in emb:
                    out.append(Probe(s, None, emb[(s, i)], {}))
                else:
                    st, q = failed[(s, i)]
                    out.append(Probe(s, st, None, q))
        return out

    v_ids, v_mat = templates(probes(v_enr, enroll_images))
    cal = calibrate(
        probes(v_enr, probe_images),
        probes(v_unk, probe_images),
        v_ids,
        v_mat,
        target_far,
        margin,
    )
    thr = cal["threshold"]
    t_ids, t_mat = templates(probes(t_enr, enroll_images))
    test = rates(
        probes(t_enr, probe_images),
        probes(t_unk, probe_images),
        t_ids,
        t_mat,
        thr,
        margin,
    )
    return {
        "status": "ejecutado",
        "model_id": embedder.model_id,
        "cloud": embedder.cloud,
        "dim": len(vecs[0]) if vecs else None,
        "subjects": {"val": len(val_subjects), "test": len(test_subjects)},
        "calibration_val": cal,
        "test": test,
        "detection_failures": len(failed),
        "images_embedded": images_embedded,
        "latency": {
            "ms_per_image_batched": batch_s * 1000 / len(vecs) if vecs else 0.0,
            "p50_single_ms": _percentile(singles, 0.5),
            "p90_single_ms": _percentile(singles, 0.9),
        },
        "cost_estimate_usd": images_embedded * price_per_image_usd,
    }


def run_comparison(
    names: Sequence[str],
    root: Path,
    *,
    subjects: int,
    per_subject: int,
    target_far: float,
    enroll_n: int = 5,
    env: Mapping[str, str] | None = None,
    gemini_min_interval: float = 0.0,
) -> dict[str, Any]:
    """Ejecuta cada embedder sobre los mismos sujetos disjuntos; Gemini solo si hay clave."""
    env = os.environ if env is None else env
    half = subjects // 2
    val, test = list(range(half)), list(range(half, subjects))
    report: dict[str, Any] = {
        "data": str(root),
        "target_far": target_far,
        "enroll_images": enroll_n,
        "probe_images": per_subject - enroll_n,
        "models": {},
    }
    for name in names:
        if name == "gemini" and not env.get("GEMINI_API_KEY"):
            report["models"][name] = skipped_report(
                "gemini-embedding-2", "GEMINI_API_KEY no definida"
            )
            continue
        try:
            engine, embedder = build_embedder(
                name, cloud_consent=True
            )  # DigiFace es sintético: no hay personas reales que consentir
            if name == "gemini" and hasattr(embedder, "min_interval"):
                embedder.min_interval = gemini_min_interval
            report["models"][name] = run_embedder(
                engine,
                embedder,
                root,
                val_subjects=val,
                test_subjects=test,
                enroll_images=range(enroll_n),
                probe_images=range(enroll_n, per_subject),
                target_far=target_far,
                price_per_image_usd=GEMINI_USD_PER_IMAGE if name == "gemini" else 0.0,
            )
        except Exception as e:  # noqa: BLE001  un fallo de un embedder no tumba el resto del informe
            report["models"][name] = skipped_report(name, f"{type(e).__name__}: {e}")
    return report
