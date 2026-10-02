"""CLI del benchmark de eventos: `uv run python scripts/events.py <comando> --help`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_events_config


def _brief(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "precision",
        "recall",
        "f1",
        "false_alarms_per_hour",
        "ttd_p50_s",
        "ttd_p90_s",
        "severity_accuracy",
    )
    return {t: {k: round(m[k], 3) for k in keys} for t, m in report["metrics"].items()}


def cmd_benchmark(a: argparse.Namespace) -> None:
    from .benchmark import run, scenarios

    zones, policy = load_events_config(a.config)
    test = scenarios(zones, "test", a.test_scenarios)
    out: dict[str, Any] = {"config": a.config, "split": "test"}
    out["rule"] = run(zones, policy, test)
    print("Regla (held-out):", json.dumps(_brief(out["rule"]), indent=1))
    if a.learned:
        from .learned import calibrate_events, collect, train

        x_tr, y_tr = collect(
            scenarios(zones, "train", a.train_scenarios), zones, policy
        )
        model = train(x_tr, y_tr)
        cal = calibrate_events(
            model, zones, policy, scenarios(zones, "val", a.val_scenarios)
        )
        model.reset()
        out["learned"] = run(zones, policy, test, decider=model)
        out["learned"]["training"] = {
            "train_samples": len(y_tr),
            "train_positive_rate": float(y_tr.mean()),
            "val_scenarios": a.val_scenarios,
            "calibration": cal,
        }
        print("Aprendido (held-out):", json.dumps(_brief(out["learned"]), indent=1))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Reporte: {a.output}")


def cmd_evaluate(a: argparse.Namespace) -> None:
    """Corre la regla sobre tracks reales etiquetados."""
    from .engine import EventEngine
    from .io import load_ground_truth, load_mot_tracks
    from .metrics import TypeStats, match_events, summarize
    from .model import EventType

    zones, policy = load_events_config(a.config)
    auth = {int(v) for v in a.authorized_ids.split(",")} if a.authorized_ids else set()
    obs = load_mot_tracks(a.tracks, a.stream_id, a.fps, a.width, a.height, auth)
    gt = load_ground_truth(a.ground_truth)
    preds = EventEngine(zones, policy).process(obs)
    stats: dict[EventType, TypeStats] = {}
    match_events(preds, gt, stats, tag=a.stream_id)
    hours = (obs[-1].t - obs[0].t) / 3600 if obs else 0.0
    report = {"tracks": a.tracks, "hours": hours, "metrics": summarize(stats, hours)}
    print(json.dumps(_brief(report), indent=1))
    if a.output:
        Path(a.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="events", description="Benchmark de merodeo, intrusión y aglomeración"
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser(
        "benchmark", help="Escenarios sintéticos etiquetados: regla vs modelo aprendido"
    )
    s.add_argument("--config", default="configs/events.toml")
    s.add_argument(
        "--test-scenarios",
        type=int,
        default=None,
        help="Por defecto, los 300 del held-out",
    )
    s.add_argument("--train-scenarios", type=int, default=None)
    s.add_argument("--val-scenarios", type=int, default=50)
    s.add_argument(
        "--learned", action="store_true", help="Entrena y evalúa el MLP de merodeo"
    )
    s.add_argument("--output", default="reports/events_benchmark.json")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser(
        "evaluate", help="Tracks reales (MOT) contra eventos etiquetados (CSV)"
    )
    s.add_argument("--config", default="configs/events.toml")
    s.add_argument("--tracks", required=True)
    s.add_argument("--ground-truth", required=True)
    s.add_argument("--stream-id", required=True)
    s.add_argument("--fps", type=float, required=True)
    s.add_argument("--width", type=int, required=True)
    s.add_argument("--height", type=int, required=True)
    s.add_argument(
        "--authorized-ids", help="IDs de track autorizados, separados por coma"
    )
    s.add_argument("--output")
    s.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
