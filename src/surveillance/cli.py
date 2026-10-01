"""CLI del detector de vigilancia: `uv run python scripts/surveillance.py <comando> --help`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .classes import NAMES
from .detector import SurveillanceDetector, load_surveillance_config

DATA = Path("datasets/surveillance")
TEST_SPLITS = ("test_coco", "test_mot", "test_simuletic")


def _dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_prepare(a: argparse.Namespace) -> None:
    from .dataset import (
        MOT_HELDOUT,
        MOT_VAL,
        coco_split,
        mot_split,
        write_yaml,
        yolo_person_only,
    )

    coco, mot, out = Path(a.coco), Path(a.mot), Path(a.out)
    ann = coco / "annotations"
    report = [
        coco_split(
            ann / "instances_train2017.json",
            coco / "train2017",
            out,
            "train_coco",
            max_images=a.coco_train,
            negatives=a.coco_train // 10,
            download=True,
        ),
        coco_split(
            ann / "instances_val2017.json",
            coco / "val2017",
            out,
            "val_coco",
            image_filter="even",
        ),
        coco_split(
            ann / "instances_val2017.json",
            coco / "val2017",
            out,
            "test_coco",
            image_filter="odd",
            negatives=a.negatives,
            negatives_split="negatives",
        ),
    ]
    seqs = sorted(p for p in (mot / "train").iterdir() if p.is_dir())
    train_seqs = [p for p in seqs if p.name not in MOT_HELDOUT and p.name != MOT_VAL]
    val_seq = [p for p in seqs if p.name == MOT_VAL]
    report.append(
        mot_split(train_seqs, out, "train_mot", frame_step=3, min_visibility=0.25)
    )
    report.append(
        mot_split(
            val_seq,
            out,
            "train_mot",
            frame_step=3,
            min_visibility=0.25,
            frame_range=(0.0, 0.7),
        )
    )
    report.append(
        mot_split(val_seq, out, "val_mot", frame_step=3, frame_range=(0.7, 1.0))
    )
    report.append(
        mot_split(
            [p for p in seqs if p.name in MOT_HELDOUT], out, "test_mot", frame_step=2
        )
    )
    if a.simuletic:
        report.append(yolo_person_only(a.simuletic, out, "test_simuletic"))
    write_yaml(
        out,
        "train",
        {"train": ["train_coco", "train_mot"], "val": ["val_coco", "val_mot"]},
    )
    _dump(out / "prepare_report.json", report)
    for r in report:
        print(
            f"{r['split']:>15}: {r['images']} imágenes, {r['boxes']} cajas, negativos={r.get('negatives', 0)}"
        )


def _perturbations() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    import cv2

    lut = (np.linspace(0, 1, 256) ** 2.5 * 255 * 0.6).astype("uint8")
    kernel = np.zeros((15, 15), dtype="float32")
    kernel[7, :] = 1 / 15
    return {
        # Poca luz: gamma 2,5 y 60 % de brillo, más ruido de sensor.
        "dark": lambda f: np.clip(
            cv2.LUT(f, lut).astype("int16")
            + np.random.default_rng(0).normal(0, 6, f.shape).astype("int16"),
            0,
            255,
        ).astype("uint8"),
        # Desenfoque de movimiento horizontal de 15 px.
        "blur": lambda f: cv2.filter2D(f, -1, kernel),
    }


def predict_split(
    det: SurveillanceDetector,
    split_dir: Path,
    perturb: str | None = None,
    batch: int = 16,
) -> dict:
    import cv2

    paths = sorted(
        p for p in split_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
    )
    fn = _perturbations()[perturb] if perturb else None
    out: dict[str, Any] = {"sizes": {}, "preds": {}}
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        frames = []
        for p in chunk:
            f = cv2.imread(str(p))
            if f is None:
                raise ValueError(f"Imagen ilegible: {p}")
            frames.append(f)
        if fn:
            frames = [fn(f) for f in frames]
        for p, f, dets in zip(chunk, frames, det.raw(frames), strict=True):
            out["sizes"][p.stem] = [f.shape[1], f.shape[0]]
            out["preds"][p.stem] = [
                [c, round(s, 4), [round(v, 1) for v in b]] for c, s, b in dets
            ]
    return out


def cmd_predict(a: argparse.Namespace) -> None:
    cfg = load_surveillance_config(a.config, model=a.model, imgsz=a.imgsz)
    det = SurveillanceDetector(cfg, min_conf=0.001)
    cache = Path(a.cache) / a.tag
    jobs = [(s, None) for s in a.splits] + [("test_mot", p) for p in a.perturb]
    for split, pert in jobs:
        name = f"{split}_{pert}" if pert else split
        res = predict_split(det, Path(a.data) / "images" / split, pert)
        _dump(cache / f"{name}.json", res)
        print(f"{name}: {len(res['preds'])} imágenes")


def _load(cache: Path, name: str) -> tuple[dict, dict]:
    d = json.loads((cache / f"{name}.json").read_text(encoding="utf-8"))
    preds = {k: [(int(c), float(s), b) for c, s, b in v] for k, v in d["preds"].items()}
    return preds, {k: tuple(v) for k, v in d["sizes"].items()}


def cmd_evaluate(a: argparse.Namespace) -> None:
    from .metrics import (
        best_f1_threshold,
        evaluate,
        load_yolo_gt,
        match_class,
        negative_fp_rate,
        person_recall_buckets,
    )

    cache, data = Path(a.cache) / a.tag, Path(a.data)

    def split(name: str, labels: str | None = None) -> tuple[dict, dict]:
        preds, sizes = _load(cache, name)
        return preds, load_yolo_gt(str(data / "labels" / (labels or name)), sizes)

    # 1) Calibración por clase en val (nunca en test).
    vc_p, vc_g = split("val_coco")
    vm_p, vm_g = split("val_mot")
    thresholds: dict[str, float] = {}
    for c, n in enumerate(NAMES):
        p, g = (dict(vc_p), dict(vc_g))
        if c == 0:
            p.update(vm_p)
            g.update(vm_g)
        thresholds[n] = best_f1_threshold(match_class(p, g, c))

    report: dict[str, Any] = {"tag": a.tag, "thresholds": thresholds, "val": {}}
    report["val"]["coco"] = evaluate(vc_p, vc_g, thresholds)
    report["val"]["mot"] = evaluate(vm_p, vm_g, thresholds, classes=[0])
    # 2) Held-out.
    report["test"] = {}
    for name in TEST_SPLITS:
        if not (cache / f"{name}.json").exists():
            continue
        p, g = split(name)
        report["test"][name] = evaluate(
            p, g, thresholds, classes=None if name == "test_coco" else [0]
        )
        if name == "test_mot":
            meta = json.loads((data / "meta" / "test_mot.json").read_text())
            report["test"][name]["person_buckets"] = person_recall_buckets(
                p, meta, thresholds["person"]
            )
    for pert in ("dark", "blur"):
        if (cache / f"test_mot_{pert}.json").exists():
            p, g = split(f"test_mot_{pert}", "test_mot")
            report["test"][f"test_mot_{pert}"] = evaluate(p, g, thresholds, classes=[0])
    if (cache / "negatives.json").exists():
        report["test"]["negatives"] = negative_fp_rate(
            _load(cache, "negatives")[0], thresholds
        )
    _dump(Path(a.output), report)
    if a.write_config:
        cfg = load_surveillance_config(a.config)
        lines = [
            f"# Generado por `surveillance.py evaluate --tag {a.tag}`: umbrales calibrados en val.",
            "[detector]",
            f'version = "{a.version}"',
            f'model = "{a.model or cfg.model}"',
            f"imgsz = {cfg.imgsz}",
            f'device = "{cfg.device}"',
            f"half = {str(cfg.half).lower()}",
            f"iou = {cfg.iou}",
            f"default_conf = {cfg.default_conf}",
            "",
            "[conf]",
            *[f"{n} = {t}" for n, t in thresholds.items()],
        ]
        Path(a.write_config).write_text("\n".join(lines) + "\n", encoding="utf-8")
    brief = {
        k: {
            "map50": round(v["map50"], 3),
            "map50_95": round(v["map50_95"], 3),
            **(
                {
                    "person_P/R": [
                        round(v["per_class"]["person"]["precision"], 3),
                        round(v["per_class"]["person"]["recall"], 3),
                    ]
                }
                if "person" in v.get("per_class", {})
                else {}
            ),
        }
        if "map50" in v
        else {"fp_image_rate": round(v["fp_image_rate"], 3)}
        for k, v in report["test"].items()
    }
    print(json.dumps({"thresholds": thresholds, "test": brief}, indent=1))


def cmd_train(a: argparse.Namespace) -> None:
    from ultralytics import YOLO

    YOLO(a.base).train(
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
        lr0=a.lr0,
        freeze=a.freeze,
    )
    print(f"Pesos: {Path(a.project) / a.name / 'weights' / 'best.pt'}")


def cmd_annotate(a: argparse.Namespace) -> None:
    from firearm.video import annotate_video

    cfg = load_surveillance_config(a.config)
    stats = annotate_video(
        SurveillanceDetector(cfg),
        a.input,
        a.output,
        weapon_classes=(),
        max_frames=a.max_frames,
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="surveillance", description="Detector de personas y objetos para CCTV"
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser(
        "prepare", help="Construye los splits YOLO (COCO, MOT16, Simuletic, negativos)"
    )
    s.add_argument("--coco", default="data/raw/coco")
    s.add_argument("--mot", default="data/raw/mot")
    s.add_argument(
        "--simuletic",
        default="data/raw/cctv-weapon/Dataset",
        help="Carpeta YOLO con images/ y labels/ (0=person)",
    )
    s.add_argument("--coco-train", type=int, default=6000)
    s.add_argument("--negatives", type=int, default=500)
    s.add_argument("--out", default=str(DATA))
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("predict", help="Cachea predicciones (conf 0,001) por split")
    s.add_argument("--config", default="configs/surveillance.toml")
    s.add_argument("--model")
    s.add_argument("--tag", required=True)
    s.add_argument("--data", default=str(DATA))
    s.add_argument("--cache", default="reports/surveillance/preds")
    s.add_argument(
        "--splits",
        nargs="*",
        default=["val_coco", "val_mot", *TEST_SPLITS, "negatives"],
    )
    s.add_argument("--perturb", nargs="*", default=["dark", "blur"])
    s.add_argument(
        "--imgsz", type=int, help="Sobrescribe el tamaño de entrada de la config"
    )
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("evaluate", help="Calibra umbrales en val y reporta el held-out")
    s.add_argument("--tag", required=True)
    s.add_argument("--config", default="configs/surveillance.toml")
    s.add_argument("--model")
    s.add_argument("--version", default="v0")
    s.add_argument("--data", default=str(DATA))
    s.add_argument("--cache", default="reports/surveillance/preds")
    s.add_argument("--output", default="reports/surveillance/eval.json")
    s.add_argument(
        "--write-config", help="Escribe un TOML versionado con los umbrales calibrados"
    )
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser(
        "train", help="Fine-tuning de YOLOv8n a las 9 clases (COCO + MOT16)"
    )
    s.add_argument("--data", default=str(DATA / "train.yaml"))
    s.add_argument("--base", default="weights/yolov8n.pt")
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--device", default="0")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--patience", type=int, default=15)
    s.add_argument("--lr0", type=float, default=0.002)
    s.add_argument("--freeze", type=int, default=0)
    s.add_argument("--project", default="runs/surveillance")
    s.add_argument("--name", default="yolov8n_coco_mot")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser(
        "annotate", help="MP4 anotado + métricas de FPS, latencia y memoria"
    )
    s.add_argument("--config", default="configs/surveillance.toml")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--max-frames", type=int)
    s.set_defaults(func=cmd_annotate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
