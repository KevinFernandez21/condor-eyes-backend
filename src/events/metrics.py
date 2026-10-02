"""Emparejamiento evento predicho ↔ etiqueta y métricas por tipo de evento."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .model import Event, EventType, GroundTruthEvent, Severity

# Margen para aceptar una detección antes del inicio etiquetado o después del fin.
EARLY_TOL_S = 2.0
LATE_TOL_S = 5.0


@dataclass
class TypeStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ttd: list[float] = field(default_factory=list)
    severity_ok: int = 0
    false_alarms: list[dict] = field(default_factory=list)
    misses: list[dict] = field(default_factory=list)


def _matches(p: Event, g: GroundTruthEvent) -> bool:
    if p.type != g.type or p.stream_id != g.stream_id or p.zone != g.zone:
        return False
    if not g.start_t - EARLY_TOL_S <= p.detect_t <= g.end_t + LATE_TOL_S:
        return False
    return not g.track_ids or bool(set(p.track_ids) & set(g.track_ids))


def expected_severity(g: GroundTruthEvent, authorized_severity: Severity) -> Severity:
    return authorized_severity if g.authorized else Severity.ALERT


def match_events(
    preds: Sequence[Event],
    gts: Sequence[GroundTruthEvent],
    stats: dict[EventType, TypeStats],
    authorized_severity: Severity = Severity.REVIEW,
    tag: str = "",
) -> tuple[list[dict], list[dict]]:
    """Emparejamiento voraz 1 a 1 en orden de detección; acumula en `stats`.

    Devuelve las falsas alarmas y las omisiones de esta llamada.
    """
    used: set[int] = set()
    fps: list[dict] = []
    fns: list[dict] = []
    for p in sorted(preds, key=lambda e: e.detect_t):
        st = stats.setdefault(p.type, TypeStats())
        cand = [i for i, g in enumerate(gts) if i not in used and _matches(p, g)]
        if not cand:
            st.fp += 1
            fps.append({"scenario": tag, **p.to_payload()})
            st.false_alarms.append(fps[-1])
            continue
        i = min(cand, key=lambda i: abs(p.detect_t - gts[i].start_t))
        used.add(i)
        g = gts[i]
        st.tp += 1
        st.ttd.append(p.detect_t - g.start_t)
        st.severity_ok += p.severity == expected_severity(g, authorized_severity)
    for i, g in enumerate(gts):
        if i not in used:
            st = stats.setdefault(g.type, TypeStats())
            st.fn += 1
            fns.append(
                {
                    "scenario": tag,
                    "type": g.type.value,
                    "start_t": g.start_t,
                    "end_t": g.end_t,
                    "track_ids": list(g.track_ids),
                    "authorized": g.authorized,
                }
            )
            st.misses.append(fns[-1])
    return fps, fns


def _pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(
    stats: dict[EventType, TypeStats], hours: float, examples: int = 5
) -> dict:
    out: dict = {}
    for et in EventType:
        st = stats.get(et, TypeStats())
        prec = st.tp / (st.tp + st.fp) if st.tp + st.fp else 0.0
        rec = st.tp / (st.tp + st.fn) if st.tp + st.fn else 0.0
        out[et.value] = {
            "tp": st.tp,
            "fp": st.fp,
            "fn": st.fn,
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "false_alarms_per_hour": st.fp / hours if hours else 0.0,
            "ttd_mean_s": sum(st.ttd) / len(st.ttd) if st.ttd else 0.0,
            "ttd_p50_s": _pct(st.ttd, 0.5),
            "ttd_p90_s": _pct(st.ttd, 0.9),
            "severity_accuracy": st.severity_ok / st.tp if st.tp else 0.0,
            "false_alarm_examples": st.false_alarms[:examples],
            "miss_examples": st.misses[:examples],
        }
    return out
