"""CLI de verificación facial: `uv run python scripts/faceid.py <comando> --help`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

STORE = "data/faceid/enrolled.json"


def _verifier(a: argparse.Namespace, store_path: str | None = None) -> Any:
    from .engine import FaceEngine
    from .verify import EnrollmentStore, Verifier, VerifyPolicy

    policy = VerifyPolicy(threshold=a.threshold, margin=a.margin)
    return Verifier(
        FaceEngine(a.embedder), EnrollmentStore(store_path or a.store), policy
    ), policy


def _add_common(s: argparse.ArgumentParser) -> None:
    s.add_argument(
        "--embedder",
        default="sface",
        choices=["sface", "sface_int8", "mobilenet_imagenet"],
    )
    s.add_argument("--threshold", type=float, default=0.65)
    s.add_argument("--margin", type=float, default=0.03)
    s.add_argument("--store", default=STORE)


def cmd_download(a: argparse.Namespace) -> None:
    from .digiface import download_subset

    n = download_subset(a.out, range(a.first, a.first + a.subjects), a.per_subject)
    print(f"{n} imágenes nuevas en {a.out}")


def cmd_benchmark(a: argparse.Namespace) -> None:
    from .benchmark import run

    root = Path(a.data)
    half = a.subjects // 2
    val, test = list(range(half)), list(range(half, a.subjects))
    report: dict[str, Any] = {
        "data": str(root),
        "val_subjects": len(val),
        "test_subjects": len(test),
        "target_far": a.target_far,
        "models": {},
    }
    for emb in a.embedders:
        a.embedder = emb
        verifier, policy = _verifier(
            a, store_path=str(Path(a.output).with_suffix(".unused.json"))
        )
        r = run(
            verifier, policy, root, val, test, range(5, a.per_subject), a.target_far
        )
        report["models"][emb] = r
        o = r["test"]["original"]
        print(
            f"{emb:>20}: umbral={r['calibration_val']['threshold']} FAR={o['far']:.4f} FRR={o['frr']:.3f} "
            f"inconcluso={o['genuine_inconclusive']:.3f} p50={r['latency']['p50_ms']:.1f} ms"
        )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Reporte: {a.output}")


def cmd_compare_embedders(a: argparse.Namespace) -> None:
    from .embed_benchmark import run_comparison

    report = run_comparison(
        a.embedders,
        Path(a.data),
        subjects=a.subjects,
        per_subject=a.per_subject,
        target_far=a.target_far,
        gemini_min_interval=a.gemini_min_interval,
    )
    for name, r in report["models"].items():
        if r["status"] != "ejecutado":
            print(f"{name:>20}: {r['status']} ({r['reason']})")
            continue
        t = r["test"]
        print(
            f"{name:>20}: umbral={r['calibration_val']['threshold']} FAR={t['far']:.4f} "
            f"FRR={t['frr']:.3f} p50={r['latency']['p50_single_ms']:.1f} ms "
            f"coste=${r['cost_estimate_usd']:.4f}"
        )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Reporte: {a.output}")


def cmd_enroll(a: argparse.Namespace) -> None:
    import cv2

    verifier, _ = _verifier(a)
    embs, rejected = [], {}
    for p in a.images:
        st, emb, _q = verifier.embed_one(cv2.imread(p))
        if emb is None:
            rejected[p] = st.value if st else "invalid_input"
        else:
            embs.append(emb)
    verifier.store.enroll(
        a.person_id, embs, a.consent_ref, a.retention_days, actor=a.actor
    )
    print(
        json.dumps(
            {"person_id": a.person_id, "samples": len(embs), "rejected": rejected},
            ensure_ascii=False,
        )
    )


def cmd_verify(a: argparse.Namespace) -> None:
    import cv2

    verifier, _ = _verifier(a)
    res = verifier.verify(cv2.imread(a.image))
    verifier.store._audit(
        "verify", status=res.status.value, person_id=res.person_id, actor=a.actor
    )
    print(json.dumps(res.to_payload(), ensure_ascii=False))


def cmd_delete(a: argparse.Namespace) -> None:
    from .verify import EnrollmentStore

    print(EnrollmentStore(a.store).delete(a.person_id, actor=a.actor))


def cmd_purge(a: argparse.Namespace) -> None:
    from .verify import EnrollmentStore

    print(EnrollmentStore(a.store).purge_expired())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="faceid",
        description="Verificación facial de personal autorizado (con consentimiento)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "download", help="Baja un subconjunto de DigiFace-1M (sintético) por HTTP Range"
    )
    s.add_argument("--out", default="data/raw/digiface")
    s.add_argument("--first", type=int, default=0)
    s.add_argument("--subjects", type=int, default=400)
    s.add_argument("--per-subject", type=int, default=24)
    s.set_defaults(func=cmd_download)

    s = sub.add_parser(
        "benchmark", help="Calibra en sujetos de val y mide FAR/FRR en sujetos de test"
    )
    _add_common(s)
    s.add_argument("--data", default="data/raw/digiface")
    s.add_argument("--subjects", type=int, default=400)
    s.add_argument("--per-subject", type=int, default=24)
    s.add_argument(
        "--embedders", nargs="+", default=["sface", "sface_int8", "mobilenet_imagenet"]
    )
    s.add_argument("--target-far", type=float, default=0.001)
    s.add_argument("--output", default="reports/faceid_benchmark.json")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser(
        "compare-embedders",
        help="SFace vs Gemini Embedding 2 vs CNN en sujetos disjuntos (Gemini solo con GEMINI_API_KEY)",
    )
    s.add_argument("--data", default="data/raw/digiface")
    s.add_argument("--subjects", type=int, default=400)
    s.add_argument("--per-subject", type=int, default=12)
    s.add_argument(
        "--embedders", nargs="+", default=["sface", "mobilenet_imagenet", "gemini"]
    )
    s.add_argument("--target-far", type=float, default=0.001)
    s.add_argument("--gemini-min-interval", type=float, default=0.0)
    s.add_argument("--output", default="reports/faceid_embedders.json")
    s.set_defaults(func=cmd_compare_embedders)

    s = sub.add_parser(
        "enroll",
        help="Enrola a una persona con consentimiento (solo guarda embeddings)",
    )
    _add_common(s)
    s.add_argument("--person-id", required=True)
    s.add_argument(
        "--consent-ref", required=True, help="Referencia al consentimiento firmado"
    )
    s.add_argument("--retention-days", type=float, default=365)
    s.add_argument("--actor", default="operador")
    s.add_argument("images", nargs="+")
    s.set_defaults(func=cmd_enroll)

    s = sub.add_parser(
        "verify", help="Devuelve evidencia de identidad (metadata) para el operador"
    )
    _add_common(s)
    s.add_argument("--actor", default="identity-agent")
    s.add_argument("image")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("delete", help="Borra la plantilla de una persona")
    s.add_argument("--store", default=STORE)
    s.add_argument("--person-id", required=True)
    s.add_argument("--actor", default="operador")
    s.set_defaults(func=cmd_delete)

    s = sub.add_parser("purge", help="Borra las plantillas caducadas")
    s.add_argument("--store", default=STORE)
    s.set_defaults(func=cmd_purge)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
