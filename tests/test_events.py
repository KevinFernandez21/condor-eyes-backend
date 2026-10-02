"""Tests del detector de eventos: reglas, autorización, contrato de bus, métricas y datos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bus import Topic
from events import (
    AuthorizedPolicy,
    Event,
    EventEngine,
    EventPolicy,
    EventType,
    GroundTruthEvent,
    Severity,
    TrackObservation,
    Zone,
    load_events_config,
    point_in_polygon,
)
from events.benchmark import SPLITS, run, scenarios
from events.io import load_ground_truth, load_mot_tracks
from events.learned import LearnedLoiter, collect, train
from events.metrics import TypeStats, match_events, summarize
from events.simulate import generate

SQUARE = ((0.5, 0.5), (0.9, 0.5), (0.9, 0.9), (0.5, 0.9))
RESTRICTED = Zone("r", "cam", SQUARE, restricted=True, intrusion_min_s=0.6)
LOITER = Zone("l", "cam", SQUARE, loiter_dwell_s=10.0, loiter_max_disp=0.1)
CROWD = Zone("c", "cam", SQUARE, crowd_threshold=3, crowd_dwell_s=2.0)
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "events.toml"


def track(tid, pts, t0=0.0, dt=0.2, authorized=False):
    return [
        TrackObservation("cam", tid, t0 + i * dt, x, y, authorized)
        for i, (x, y) in enumerate(pts)
    ]


def still(x, y, seconds, dt=0.2):
    return [(x, y)] * int(seconds / dt)


# --- geometría ---------------------------------------------------------------


def test_point_in_polygon():
    assert point_in_polygon(0.7, 0.7, SQUARE)
    assert not point_in_polygon(0.3, 0.7, SQUARE)
    assert not point_in_polygon(0.7, 0.95, SQUARE)


def test_zone_validation():
    with pytest.raises(ValueError):
        Zone("x", "cam", ((0, 0), (1, 1)))
    with pytest.raises(ValueError):
        Zone("x", "cam", SQUARE, crowd_threshold=1)


# --- intrusión -----------------------------------------------------------------


def test_intrusion_fires_once_after_debounce():
    obs = track(1, still(0.3, 0.7, 2) + still(0.7, 0.7, 3))
    events = EventEngine([RESTRICTED]).process(obs)
    assert len(events) == 1
    e = events[0]
    assert e.type is EventType.INTRUSION and e.severity is Severity.ALERT
    assert e.start_t == pytest.approx(2.0) and e.detect_t == pytest.approx(2.6)
    assert e.evidence["observations_inside"] >= 3


def test_border_flicker_does_not_fire():
    pts = [(0.51 if i % 2 else 0.49, 0.7) for i in range(100)]
    assert EventEngine([RESTRICTED]).process(track(1, pts)) == []


def test_intrusion_rearms_only_after_exit_grace():
    pts = (
        still(0.7, 0.7, 2)
        + still(0.3, 0.7, 0.4)
        + still(0.7, 0.7, 2)
        + still(0.3, 0.7, 2)
        + still(0.7, 0.7, 2)
    )
    events = EventEngine([RESTRICTED]).process(track(1, pts))
    # La salida breve (< exit_grace_s) no re-arma; la larga sí.
    assert len(events) == 2


# --- merodeo -------------------------------------------------------------------


def test_loitering_fires_after_dwell_with_small_displacement():
    events = EventEngine([LOITER]).process(track(1, still(0.7, 0.7, 15)))
    assert [e.type for e in events] == [EventType.LOITERING]
    assert events[0].detect_t == pytest.approx(10.0, abs=0.21)
    assert events[0].evidence["net_displacement"] < 0.1


def test_slow_transit_is_not_loitering():
    pts = [(0.5 + 0.4 * i / 100, 0.7) for i in range(100)]  # 20 s cruzando la zona
    assert EventEngine([LOITER]).process(track(1, pts)) == []


def test_short_occlusion_keeps_dwell_and_long_one_resets_it():
    short = track(1, still(0.7, 0.7, 6)) + track(1, still(0.7, 0.7, 6), t0=7.0)
    assert len(EventEngine([LOITER]).process(short)) == 1
    long_gap = track(1, still(0.7, 0.7, 6)) + track(1, still(0.7, 0.7, 6), t0=10.0)
    assert EventEngine([LOITER]).process(long_gap) == []


# --- autorización ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "severity"),
    [
        (AuthorizedPolicy.DOWNGRADE, Severity.REVIEW),
        (AuthorizedPolicy.SUPPRESS, Severity.SUPPRESSED),
    ],
)
def test_authorized_person_downgrades_without_hiding_evidence(policy, severity):
    obs = track(1, still(0.7, 0.7, 3), authorized=True)
    events = EventEngine([RESTRICTED], EventPolicy(authorized=policy)).process(obs)
    assert len(events) == 1
    assert events[0].severity is severity and events[0].authorized
    assert events[0].evidence["inside_s"] > 0


# --- aglomeración ----------------------------------------------------------------


def _crowd(n):
    obs = [o for i in range(n) for o in track(i, still(0.6 + 0.05 * i, 0.7, 4))]
    return sorted(obs, key=lambda o: (o.t, o.track_id))


def test_crowding_needs_threshold_sustained():
    events = EventEngine([CROWD]).process(_crowd(3))
    assert [e.type for e in events] == [EventType.CROWDING]
    assert events[0].track_ids == (0, 1, 2)
    assert EventEngine([CROWD]).process(_crowd(2)) == []


def test_other_stream_is_ignored():
    obs = [TrackObservation("otra", 1, i * 0.2, 0.7, 0.7) for i in range(20)]
    assert EventEngine([RESTRICTED]).process(obs) == []


# --- contrato de bus: solo metadata ---------------------------------------------


def test_event_envelope_is_metadata_only():
    e = EventEngine([RESTRICTED]).process(track(1, still(0.7, 0.7, 3)))[0]
    env = e.to_envelope()
    assert env.source == "event" and env.stream_id == "cam"
    payload = json.loads(json.dumps(env.payload))  # serializable sin pérdida
    assert payload["type"] == "zone_intrusion" and payload["track_ids"] == [1]

    def primitives(v):
        if isinstance(v, dict):
            return all(primitives(x) for x in v.values())
        if isinstance(v, list):
            return all(primitives(x) for x in v)
        return isinstance(v, (str, int, float, bool)) or v is None

    assert primitives(env.payload)
    assert Topic.EVENTS.value == "events"


# --- métricas -----------------------------------------------------------------------


def _ev(t, tids=(1,), etype=EventType.INTRUSION, sev=Severity.ALERT):
    return Event(etype, "cam", "r", tids, t - 1, t, 1.0, sev, False)


def test_match_events_counts_tp_fp_fn_and_ttd():
    gts = [
        GroundTruthEvent(EventType.INTRUSION, "cam", "r", 10.0, 20.0, (1,)),
        GroundTruthEvent(EventType.INTRUSION, "cam", "r", 50.0, 60.0, (2,)),
    ]
    preds = [_ev(10.8), _ev(11.0), _ev(30.0, (3,))]
    stats: dict[EventType, TypeStats] = {}
    fps, fns = match_events(preds, gts, stats)
    st = stats[EventType.INTRUSION]
    assert (st.tp, st.fp, st.fn) == (1, 2, 1)
    assert st.ttd == [pytest.approx(0.8)]
    assert len(fps) == 2 and fns[0]["start_t"] == 50.0
    s = summarize(stats, hours=0.5)["zone_intrusion"]
    assert s["false_alarms_per_hour"] == 4.0 and s["precision"] == pytest.approx(1 / 3)


def test_match_requires_track_overlap_for_track_events():
    gts = [GroundTruthEvent(EventType.INTRUSION, "cam", "r", 10.0, 20.0, (1,))]
    stats: dict[EventType, TypeStats] = {}
    match_events([_ev(11.0, (9,))], gts, stats)
    assert stats[EventType.INTRUSION].tp == 0


# --- simulador, benchmark y modelo aprendido -----------------------------------------


def test_repo_config_and_simulation_are_deterministic():
    zones, policy = load_events_config(CONFIG)
    a, b = generate(zones, 7), generate(zones, 7)
    assert a.observations == b.observations and a.ground_truth == b.ground_truth
    assert policy.authorized is AuthorizedPolicy.DOWNGRADE


def test_splits_are_disjoint():
    sets = [set(v) for v in SPLITS.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_benchmark_runs_and_reports_all_metrics():
    zones, policy = load_events_config(CONFIG)
    r = run(zones, policy, scenarios(zones, "val", 3))
    assert r["hours"] == pytest.approx(3 * 180 / 3600)
    assert set(r["metrics"]) == {"loitering", "zone_intrusion", "crowding"}
    for m in r["metrics"].values():
        assert {"precision", "recall", "false_alarms_per_hour", "ttd_p50_s"} <= m.keys()
    assert r["latency_us_per_observation"]["p50"] > 0


def test_learned_decider_smooths_over_consecutive_samples():
    zones, policy = load_events_config(CONFIG)
    x, y = collect(scenarios(zones, "train", 4), zones, policy)
    assert x.shape[1] == 5 and len(x) == len(y)
    model = train(x, y, epochs=20)
    always = LearnedLoiter(
        np.zeros((5, 1)), np.zeros(1), np.zeros(1), 10.0, threshold=0.5, consecutive=3
    )
    from events.engine import LoiterFeatures

    f = LoiterFeatures(1.0, 1.0, 0.0, 0.0, key=("cam", 1, "l"))
    assert [always(f, LOITER)[0] for _ in range(3)] == [False, False, True]
    assert 0.0 <= model(f, LOITER)[1] <= 1.0


# --- datos reales ---------------------------------------------------------------------


def test_loaders_for_real_data(tmp_path: Path):
    mot = tmp_path / "t.txt"
    mot.write_text(
        "1,4,100,200,20,40,1,-1,-1,-1\n2,4,102,200,20,40,1,-1,-1,-1\n", encoding="utf-8"
    )
    obs = load_mot_tracks(mot, "cam", fps=10, width=200, height=400, authorized_ids={4})
    assert obs[0].x == pytest.approx(0.55) and obs[0].y == pytest.approx(0.6)
    assert obs[1].t == pytest.approx(0.1) and obs[0].authorized

    gt = tmp_path / "gt.csv"
    gt.write_text(
        "stream_id,event_type,zone,start_s,end_s,track_ids,authorized\n"
        "cam,loitering,plaza,30,50,4;7,no\ncam,crowding,plaza,5,9,,sí\n",
        encoding="utf-8",
    )
    rows = load_ground_truth(gt)
    assert rows[0].track_ids == (4, 7) and not rows[0].authorized
    assert (
        rows[1].type is EventType.CROWDING
        and rows[1].track_ids == ()
        and rows[1].authorized
    )
