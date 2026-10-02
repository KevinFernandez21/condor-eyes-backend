"""Benchmark del detector de eventos sobre escenarios sintéticos etiquetados."""

from __future__ import annotations

import time
import tracemalloc
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .engine import EventEngine, LoiterDecider, rule_loiter
from .metrics import TypeStats, match_events, summarize
from .model import AuthorizedPolicy, EventPolicy, EventType, Severity, Zone
from .simulate import TEST, TRAIN, Perturbation, Scenario, generate

# Semillas disjuntas por split: el held-out nunca se usa para ajustar umbrales.
SPLITS = {"train": range(300), "val": range(1000, 1100), "test": range(5000, 5300)}


def scenarios(
    zones: Sequence[Zone], split: str, n: int | None = None
) -> list[Scenario]:
    seeds = list(SPLITS[split])[:n]
    pert: Perturbation = TEST if split == "test" else TRAIN
    return [generate(zones, s, pert) for s in seeds]


def _overlaps(a: float, b: float, spans: Sequence[tuple[float, float]]) -> bool:
    return any(s <= b and a <= e for s, e in spans)


def failure_causes(item: dict, sc: Scenario) -> list[str]:
    """Etiqueta una falsa alarma u omisión con el comportamiento y las perturbaciones."""
    a = item["start_t"] - 2.0
    b = item.get("detect_t", item.get("end_t", a))
    causes: set[str] = set()
    for tid in item["track_ids"]:
        info = sc.track_info.get(tid)
        if not info:
            continue
        causes.add(f"behavior:{info['behavior']}")
        if _overlaps(a, b, info["occlusions"]):
            causes.add("occlusion")
        if info["id_switch_t"] is not None and a - 5 <= info["id_switch_t"] <= b + 5:
            causes.add("id_switch")
        if info["authorized"]:
            causes.add("authorized")
    if _overlaps(a, b, sc.shakes):
        causes.add("camera_shake")
    if item["type"] == EventType.CROWDING.value:
        causes.add("group")
    return sorted(causes) or ["unknown"]


def run(
    zones: Sequence[Zone],
    policy: EventPolicy,
    scs: Sequence[Scenario],
    decider: LoiterDecider = rule_loiter,
) -> dict[str, Any]:
    stats: dict[EventType, TypeStats] = {}
    fp_causes: dict[str, Counter[str]] = {}
    fn_causes: dict[str, Counter[str]] = {}
    lat_us: list[float] = []
    n_obs = 0
    auth_sev = (
        Severity.SUPPRESSED
        if policy.authorized is AuthorizedPolicy.SUPPRESS
        else Severity.REVIEW
    )
    tracemalloc.start()
    for sc in scs:
        reset = getattr(decider, "reset", None)
        if callable(reset):
            reset()  # los decisores con estado no deben arrastrar tracks entre escenarios
        engine = EventEngine(zones, policy, loiter_decider=decider)
        preds = []
        for obs in sc.observations:
            s = time.perf_counter()
            preds.extend(engine.update(obs))
            lat_us.append((time.perf_counter() - s) * 1e6)
        n_obs += len(sc.observations)
        fps, fns = match_events(
            preds, sc.ground_truth, stats, auth_sev, tag=str(sc.seed)
        )
        for item, bucket in [(i, fp_causes) for i in fps] + [
            (i, fn_causes) for i in fns
        ]:
            c = bucket.setdefault(item["type"], Counter())
            c.update(failure_causes(item, sc))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    hours = sum(sc.duration_s for sc in scs) / 3600.0
    lat = sorted(lat_us)
    behaviors: Counter[str] = Counter()
    gt_counts: Counter[str] = Counter()
    for sc in scs:
        behaviors.update(sc.behaviors)
        gt_counts.update(g.type.value for g in sc.ground_truth)
    return {
        "scenarios": len(scs),
        "hours": hours,
        "observations": n_obs,
        "behaviors": dict(behaviors),
        "ground_truth_events": dict(gt_counts),
        "metrics": summarize(stats, hours),
        "false_alarm_causes": {k: dict(v) for k, v in fp_causes.items()},
        "miss_causes": {k: dict(v) for k, v in fn_causes.items()},
        "latency_us_per_observation": {
            "p50": lat[len(lat) // 2] if lat else 0.0,
            "p90": lat[int(0.9 * len(lat))] if lat else 0.0,
            "p99": lat[int(0.99 * len(lat))] if lat else 0.0,
        },
        "peak_python_alloc_mb": peak / 2**20,
    }
