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
from .dataset import IMAGE_EXTS, prepare_cctv_gun


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def cmd_prepare(a: argparse.Namespace) -> None:
    report = prepare_cctv_gun(a.root, a.out, pair=a.pair, heldout=a.heldout, negatives=a.negatives)
    for s in report["splits"]:
        print(f"{s['split']:>10}: {s['images']} imágenes, {s['boxes']} cajas, faltantes={s.get('missing', 0)}")
    print(f"data.yaml en {Path(a.out) / 'data.yaml'}")


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
        data=a.data, split=a.split, imgsz=cfg.imgsz, batch=16, device=cfg.device,
        quantize=cfg.precision, plots=False, verbose=False,
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

    cfg = load_config(a.config, model=a.model, conf=a.conf, iou=a.iou, imgsz=a.imgsz, device=a.device)
    if not Path(cfg.model).exists():
        sys.exit(f"No existe el modelo {cfg.model}. Entrena con 'train' o pasa --model.")

    def progress(i: int, total: int) -> None:
        if i % 50 == 0 or i == total:
            print(f"\r{i}/{total or '?'} frames", end="", flush=True)

    stats = annotate_video(
        FirearmDetector(cfg), a.input, a.output, cfg.weapon_classes,
        max_frames=a.max_frames, progress=progress,
    )
    print()
    _print_json({k: v for k, v in stats.items() if k != "config"})
    print(f"Video anotado: {a.output}  métricas: {Path(a.output).with_suffix('.json')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="firearm", description="Detección de armas en video grabado")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare-data", help="Convierte CCTV-Gun (COCO) a formato YOLO")
    s.add_argument("--root", required=True, help="Carpeta data/ de CCTV-Gun (con all_images/)")
    s.add_argument("--out", default="datasets/cctv_gun_mgd_usrt")
    s.add_argument("--pair", default="mgd_usrt", help="Par de entrenamiento (carpeta en data/)")
    s.add_argument("--heldout", default="ucf", help="Fuente reservada como test entre dominios")
    s.add_argument("--negatives", help="Carpeta de imágenes sin arma (hard negatives)")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train", help="Fine-tuning de YOLOv8n en la GPU de la laptop")
    s.add_argument("--data", default="datasets/cctv_gun_mgd_usrt/data.yaml")
    s.add_argument("--base", default="yolov8n.pt", help="Checkpoint inicial (COCO preentrenado)")
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--batch", type=int, default=32, help="Batch de entrenamiento (la inferencia es batch 1)")
    s.add_argument("--device", default="0")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--patience", type=int, default=25)
    s.add_argument("--project", default="runs/firearm")
    s.add_argument("--name", default="yolov8n_mgd_usrt")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eval", help="mAP en el split held-out y falsos positivos en negativos")
    s.add_argument("--config", default="configs/firearm.toml")
    s.add_argument("--data", default="datasets/cctv_gun_mgd_usrt/data.yaml")
    s.add_argument("--split", default="test")
    s.add_argument("--model")
    s.add_argument("--device")
    s.add_argument("--negatives", help="Carpeta de hard negatives para medir falsos positivos")
    s.add_argument("--output", default="reports/eval_test.json")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("annotate", help="Anota un video y escribe un MP4 + métricas JSON")
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
