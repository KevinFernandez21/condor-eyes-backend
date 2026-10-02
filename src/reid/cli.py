"""CLI del benchmark de re-ID: `uv run python scripts/reid.py <comando> --help`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .market import Crop, build_splits


def _embed_all(
    embed: Callable[[np.ndarray], np.ndarray], crops: Sequence[Crop], chunk: int = 2048
) -> np.ndarray:
    """Lee y embebe por bloques para no tener toda la galería en memoria."""
    from .embed import read_crops

    out = [
        embed(read_crops([c.path for c in crops[i : i + chunk]]))
        for i in range(0, len(crops), chunk)
    ]
    return np.concatenate(out).astype("float32")


def cmd_train(a: argparse.Namespace) -> None:
    from .embed import read_crops, train_reid

    sp = build_splits(a.market)
    crops = read_crops([c.path for c in sp.train])
    out = train_reid(
        a.arch,
        crops,
        [c.pid for c in sp.train],
        Path(a.out) / f"{a.arch}_market.pt",
        epochs=a.epochs,
    )
    print(f"Pesos: {out}")


def cmd_benchmark(a: argparse.Namespace) -> None:
    import torch

    from .benchmark import association, calibrate, latency, retrieval
    from .embed import CNNEmbedder, ColorEmbedder, read_crops

    sp = build_splits(a.market)
    ids = sp.identities()
    assert not (
        ids["train"] & ids["val"]
        or ids["train"] & ids["test"]
        or ids["val"] & ids["test"]
    )
    report: dict[str, Any] = {
        "identities": {k: len(v) for k, v in ids.items()},
        "images": {
            "train": len(sp.train),
            "val_query": len(sp.val_query),
            "val_gallery": len(sp.val_gallery),
            "test_query": len(sp.test_query),
            "test_gallery": len(sp.test_gallery),
        },
        "models": {},
    }
    models: list[tuple[str, Callable[[], Any]]] = [("color-hsv", ColorEmbedder)]
    for arch in a.archs:
        models.append((f"{arch}-imagenet", partial(CNNEmbedder, arch)))
        w = Path(a.weights) / f"{arch}_market.pt"
        if w.exists():
            models.append((f"{arch}-reid", partial(CNNEmbedder, arch, w)))
    sample = read_crops([c.path for c in sp.test_query[:256]])
    for name, factory in models:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        emb = factory()
        vq, vg = _embed_all(emb, sp.val_query), _embed_all(emb, sp.val_gallery)
        tq, tg = _embed_all(emb, sp.test_query), _embed_all(emb, sp.test_gallery)
        cal = calibrate(
            vq, sp.val_query, vg, sp.val_gallery, max_false_link=a.max_false_link
        )
        from .associate import AssociatorConfig

        cfg = AssociatorConfig(threshold=cal["threshold"], margin=cal["margin"])
        r: dict[str, Any] = {
            "retrieval_val": retrieval(vq, sp.val_query, vg, sp.val_gallery),
            "retrieval_test": retrieval(tq, sp.test_query, tg, sp.test_gallery),
            "calibration_val": {
                k: cal[k]
                for k in (
                    "threshold",
                    "margin",
                    "meets_target",
                    "correct_link_rate",
                    "false_link_rate",
                )
            },
            "association_test": association(
                tq, sp.test_query, tg, sp.test_gallery, cfg
            ),
            "embedding_dim": int(tq.shape[1]),
            "latency": latency(emb, sample),
        }
        if hasattr(emb, "params"):
            r["params_m"] = emb.params / 1e6
            r["fp16_weights_mb"] = emb.params * 2 / 2**20
        report["models"][name] = r
        rt, at = r["retrieval_test"], r["association_test"]
        print(
            f"{name:>24}: rank1={rt['rank1']:.3f} mAP={rt['map']:.3f} | enlace correcto={at['correct_link_rate']:.3f} "
            f"falso={at['false_link_rate']:.3f} perdido={at['missed_link_rate']:.3f} | thr={cal['threshold']} m={cal['margin']}"
        )
        del emb
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Reporte: {a.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reid", description="Re-ID entre cámaras: benchmark y asociación"
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser(
        "train", help="Fine-tuning re-ID en las identidades de train de Market-1501"
    )
    s.add_argument("--market", default="data/raw/market/Market-1501-v15.09.15")
    s.add_argument(
        "--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small"]
    )
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--out", default="weights/reid")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser(
        "benchmark", help="Rank-1/mAP y asociación con estado inconcluso"
    )
    s.add_argument("--market", default="data/raw/market/Market-1501-v15.09.15")
    s.add_argument("--archs", nargs="+", default=["mobilenet_v3_small", "resnet18"])
    s.add_argument("--weights", default="weights/reid")
    s.add_argument("--max-false-link", type=float, default=0.01)
    s.add_argument("--output", default="reports/reid_benchmark.json")
    s.set_defaults(func=cmd_benchmark)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
