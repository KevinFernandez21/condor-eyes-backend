"""CLI del prototipo: preparar datos, entrenar, evaluar y anotar video.

Uso: `uv run python scripts/firearm.py <comando> --help`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .dataset import IMAGE_EXTS, prepare_cctv_gun, split_yolo_by_scene


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def cmd_prepare(a: argparse.Namespace) -> None:
    report = prepare_cctv_gun(
        a.root, a.out, pair=a.pair, heldout=a.heldout, negatives=a.negatives
    )
    for s in report["splits"]:
        print(
            f"{s['split']:>10}: {s['images']} imágenes, {s['boxes']} cajas, faltantes={s.get('missing', 0)}"
        )
    print(f"data.yaml en {Path(a.out) / 'data.yaml'}")


def cmd_split(a: argparse.Namespace) -> None:
    report = split_yolo_by_scene(a.src, a.out, a.names, a.val_scenes, a.test_scenes)
    for s in report["splits"]:
        print(
            f"{s['split']:>6}: {s['images']} imágenes, {s['boxes']} cajas {s['per_class']} escenas={s['scenes']}"
        )
    print(f"data.yaml en {Path(a.out) / 'data.yaml'}")


def cmd_openimages(a: argparse.Namespace) -> None:
    from .openimages import build_split, summary, write_combined_yaml

    ann, out = Path(a.ann_dir), Path(a.out)
    report = [
        build_split(
            ann / "train-bbox-filtered.csv",
            "train",
            out,
            "train",
            a.train,
            a.train_negatives,
        ),
        build_split(
            ann / "validation-bbox-filtered.csv",
            "validation",
            out,
            "val",
            a.val,
            a.val_negatives,
        ),
        build_split(
            ann / "test-bbox-filtered.csv",
            "test",
            out,
            "test",
            a.test,
            a.test_negatives,
            negatives_split="negatives",
        ),
    ]
    from .openimages import add_pseudo_persons

    # Solo train y val: test y negativos quedan con las etiquetas originales.
    pseudo = add_pseudo_persons(out, ["train", "val"])
    print(f"Personas añadidas con pseudo-etiquetas: {pseudo}")
    report.append({"pseudo_person_boxes": pseudo})
    summary(out / "prepare_report.json", report)
    for r in report:
        print(
            f"{r['split']:>6}: {r['weapon_images']} con arma ({r['weapon_boxes']} cajas), "
            f"{r['negative_images']} negativos, {r['failed']} fallidas"
        )
    root = out.parent
    rel = out.name
    sim = Path(a.simuletic).name
    write_combined_yaml(
        root,
        "firearm_v2",
        {
            "train": [f"{sim}/images/train", f"{rel}/images/train"],
            "val": [f"{sim}/images/val", f"{rel}/images/val"],
        },
    )
    write_combined_yaml(
        root,
        "firearm_v2_test_oi",
        {
            "train": [f"{rel}/images/train"],
            "val": [f"{rel}/images/val"],
            "test": [f"{rel}/images/test"],
        },
    )
    print(f"data YAML: {root / 'firearm_v2.yaml'}")


def _variant(spec: str) -> tuple[str, Any]:
    """`nombre:tipo:modelo` con tipo `single` o `twostage`."""
    from .detector import FirearmDetector
    from .twostage import TwoStageConfig, TwoStageDetector

    name, kind, model = spec.split(":", 2)
    if kind == "single":
        det: Any = FirearmDetector(
            load_config("configs/firearm.toml", model=model, conf=0.05)
        )
    elif kind == "twostage":
        det = TwoStageDetector(TwoStageConfig(weapon_model=model))
    else:
        raise ValueError(f"Tipo desconocido: {kind}")
    return name, det


def cmd_compare(a: argparse.Namespace) -> None:
    from .evaluate import (
        calibrate,
        dump,
        load_split,
        metrics,
        negatives_rate,
        predict,
        video_frames,
    )

    sim, oi = Path(a.simuletic), Path(a.openimages)
    val = load_split(sim, "val") + load_split(oi, "val")
    tests = {
        "simuletic_scene6": load_split(sim, "test"),
        "openimages_test": load_split(oi, "test"),
    }
    negatives = load_split(oi, "negatives")
    report: dict[str, Any] = {}
    for spec in a.variants:
        name, det = _variant(spec)
        vp = predict(det.infer, val)
        thr = calibrate(vp, [g for _, g in val])
        r: dict[str, Any] = {
            "spec": spec,
            "threshold_val": thr,
            "val": metrics(vp, [g for _, g in val], thr),
        }
        for tname, items in tests.items():
            r[tname] = metrics(predict(det.infer, items), [g for _, g in items], thr)
        r["negatives"] = negatives_rate(predict(det.infer, negatives), thr)
        hits = video_frames(det.infer, a.video, thr)
        r["video"] = {
            "weapon_frames": len(hits),
            "before_39": sum(i < 39 for i in hits),
            "first": hits[0] if hits else None,
            "visible_hits": sum(i >= 39 for i in hits),
        }
        report[name] = r
        det.close()
        print(
            f"{name:>18} thr={thr:.3f} | Scene6 AP50={r['simuletic_scene6']['ap50']:.3f} "
            f"P/R={r['simuletic_scene6']['precision']:.2f}/{r['simuletic_scene6']['recall']:.2f} | "
            f"OI AP50={r['openimages_test']['ap50']:.3f} P/R={r['openimages_test']['precision']:.2f}/"
            f"{r['openimages_test']['recall']:.2f} | neg={r['negatives']['rate']:.3f} | "
            f"video={r['video']['visible_hits']}/106 (falsas {r['video']['before_39']})"
        )
    dump(a.output, report)


def cmd_train(a: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(a.base)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        patience=a.patience,
        project=str(Path(a.project).resolve()),
        name=a.name,
        exist_ok=True,
        seed=0,
        deterministic=True,
        scale=a.scale,
    )
    print(f"Pesos: {Path(a.project) / a.name / 'weights' / 'best.pt'}")


def _negative_false_positives(cfg: Any, negatives: Path) -> dict[str, Any]:
    """Cuenta imágenes sin arma en las que el detector reporta alguna clase de arma."""
    import cv2

    from .detector import FirearmDetector

    det = FirearmDetector(cfg)
    files = sorted(p for p in negatives.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    hits: list[dict] = []
    try:
        for p in files:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            weapons = [d for d in det.infer(frame) if d["label"] in cfg.weapon_classes]
            if weapons:
                hits.append({"image": p.name, "max_conf": weapons[0]["conf"]})
    finally:
        det.close()
    return {
        "negatives": len(files),
        "false_positive_images": len(hits),
        "false_positive_rate": len(hits) / len(files) if files else 0.0,
        "hits": hits,
    }


def cmd_eval(a: argparse.Namespace) -> None:
    from ultralytics import YOLO

    cfg = load_config(a.config, model=a.model, device=a.device)
    out: dict[str, Any] = {"model": cfg.model, "data": a.data, "split": a.split}
    metrics = YOLO(cfg.model).val(
        data=a.data,
        split=a.split,
        imgsz=cfg.imgsz,
        batch=16,
        device=cfg.device,
        quantize=cfg.precision,
        plots=False,
        verbose=False,
    )
    names = metrics.names
    out["map50"] = float(metrics.box.map50)
    out["map50_95"] = float(metrics.box.map)
    out["per_class"] = {
        names[int(c)]: {
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "map50": float(metrics.box.ap50[i]),
            "map50_95": float(metrics.box.ap[i]),
        }
        for i, c in enumerate(metrics.box.ap_class_index)
    }
    if a.negatives:
        out["hard_negatives"] = _negative_false_positives(cfg, Path(a.negatives))
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    _print_json(out)


def cmd_annotate(a: argparse.Namespace) -> None:
    from .detector import FirearmDetector
    from .video import annotate_video

    cfg = load_config(
        a.config, model=a.model, conf=a.conf, iou=a.iou, imgsz=a.imgsz, device=a.device
    )
    if not Path(cfg.model).exists():
        sys.exit(
            f"No existe el modelo {cfg.model}. Entrena con 'train' o pasa --model."
        )

    def progress(i: int, total: int) -> None:
        if i % 50 == 0 or i == total:
            print(f"\r{i}/{total or '?'} frames", end="", flush=True)

    stats = annotate_video(
        FirearmDetector(cfg),
        a.input,
        a.output,
        cfg.weapon_classes,
        max_frames=a.max_frames,
        progress=progress,
    )
    print()
    _print_json({k: v for k, v in stats.items() if k != "config"})
    print(f"Video anotado: {a.output}  métricas: {Path(a.output).with_suffix('.json')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="firearm", description="Detección de armas en video grabado"
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare-data", help="Convierte CCTV-Gun (COCO) a formato YOLO")
    s.add_argument(
        "--root", required=True, help="Carpeta data/ de CCTV-Gun (con all_images/)"
    )
    s.add_argument("--out", default="datasets/cctv_gun_mgd_usrt")
    s.add_argument(
        "--pair", default="mgd_usrt", help="Par de entrenamiento (carpeta en data/)"
    )
    s.add_argument(
        "--heldout", default="ucf", help="Fuente reservada como test entre dominios"
    )
    s.add_argument("--negatives", help="Carpeta de imágenes sin arma (hard negatives)")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser(
        "split-scenes", help="Divide un dataset YOLO en train/val/test por escena"
    )
    s.add_argument(
        "--src",
        default="data/raw/cctv-weapon/Dataset",
        help="Carpeta con images/ y labels/",
    )
    s.add_argument("--out", default="datasets/simuletic_cctv_weapon")
    s.add_argument(
        "--names",
        nargs="+",
        default=["person", "weapon"],
        help="Nombres de clase por índice",
    )
    s.add_argument("--val-scenes", nargs="+", default=["Scene5"])
    s.add_argument("--test-scenes", nargs="+", default=["Scene6"])
    s.set_defaults(func=cmd_split)

    s = sub.add_parser(
        "compare",
        help="Compara variantes (una etapa / dos etapas) con el mismo evaluador",
    )
    s.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="nombre:single|twostage:ruta_modelo",
    )
    s.add_argument("--simuletic", default="datasets/simuletic_cctv_weapon")
    s.add_argument("--openimages", default="datasets/openimages_firearm")
    s.add_argument("--video", default="data/raw/cctv-weapon/evaluation.mp4")
    s.add_argument("--output", default="reports/firearm_cmp/compare.json")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser(
        "openimages",
        help="Arma el dataset Open Images V7 (armas + negativos difíciles)",
    )
    s.add_argument(
        "--ann-dir",
        default="data/raw/openimages",
        help="CSVs de cajas ya filtrados por clase",
    )
    s.add_argument("--out", default="datasets/openimages_firearm")
    s.add_argument("--simuletic", default="datasets/simuletic_cctv_weapon")
    s.add_argument("--train", type=int, default=5000)
    s.add_argument("--train-negatives", type=int, default=1500)
    s.add_argument("--val", type=int, default=1000)
    s.add_argument("--val-negatives", type=int, default=300)
    s.add_argument("--test", type=int, default=1000)
    s.add_argument("--test-negatives", type=int, default=500)
    s.set_defaults(func=cmd_openimages)

    s = sub.add_parser("train", help="Fine-tuning de YOLOv8n en la GPU de la laptop")
    s.add_argument("--data", default="datasets/simuletic_cctv_weapon/data.yaml")
    s.add_argument(
        "--base", default="yolov8n.pt", help="Checkpoint inicial (COCO preentrenado)"
    )
    s.add_argument("--epochs", type=int, default=150)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch de entrenamiento (la inferencia es batch 1)",
    )
    s.add_argument("--device", default="0")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--patience", type=int, default=50)
    s.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Rango de escala del aumento (0,9 = objetos más pequeños)",
    )
    s.add_argument("--project", default="runs/firearm")
    s.add_argument("--name", default="yolov8n_simuletic")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser(
        "eval", help="mAP en el split held-out y falsos positivos en negativos"
    )
    s.add_argument("--config", default="configs/firearm.toml")
    s.add_argument("--data", default="datasets/simuletic_cctv_weapon/data.yaml")
    s.add_argument("--split", default="test")
    s.add_argument("--model")
    s.add_argument("--device")
    s.add_argument(
        "--negatives", help="Carpeta de hard negatives para medir falsos positivos"
    )
    s.add_argument("--output", default="reports/eval_test.json")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser(
        "annotate", help="Anota un video y escribe un MP4 + métricas JSON"
    )
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--config", default="configs/firearm.toml")
    s.add_argument("--model")
    s.add_argument("--conf", type=float)
    s.add_argument("--iou", type=float)
    s.add_argument("--imgsz", type=int)
    s.add_argument("--device")
    s.add_argument("--max-frames", type=int)
    s.set_defaults(func=cmd_annotate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
