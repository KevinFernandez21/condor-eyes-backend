"""Línea de tiempo: cubetas por minuto y clase, acotada."""

from __future__ import annotations

from dashboard.timeline import Timeline

T0 = 1_700_000_040.0  # minuto exacto (múltiplo de 60)


def test_bucket_keeps_max_simultaneous_count_per_class_in_the_minute():
    tl = Timeline()
    tl.add("c1", T0 + 1, {"person": 1})
    tl.add("c1", T0 + 20, {"person": 3, "vehicle": 1})
    tl.add("c1", T0 + 40, {"person": 2})
    snap = tl.snapshot(T0 + 50, minutes=10)
    (bucket,) = snap["series"]["c1"]
    assert (bucket["person"], bucket["vehicle"], bucket["motion"]) == (3, 1, 0)
    assert bucket["t"] == T0


def test_series_are_ascending_and_only_include_minutes_with_data():
    tl = Timeline()
    tl.add("c1", T0 + 125, {"person": 1})
    tl.add("c1", T0 + 5, {"motion": 1})
    times = [b["t"] for b in tl.snapshot(T0 + 130, minutes=10)["series"]["c1"]]
    assert times == [T0, T0 + 120]


def test_all_series_sums_cameras_per_minute():
    tl = Timeline()
    tl.add("c1", T0 + 1, {"person": 2})
    tl.add("c2", T0 + 2, {"person": 1, "vehicle": 1})
    (bucket,) = tl.snapshot(T0 + 10, minutes=5)["series"]["all"]
    assert (bucket["person"], bucket["vehicle"]) == (3, 1)


def test_snapshot_window_excludes_old_minutes_and_reports_range():
    tl = Timeline()
    tl.add("c1", T0 - 3600, {"person": 1})
    tl.add("c1", T0, {"person": 1})
    snap = tl.snapshot(T0 + 30, minutes=10)
    assert [b["t"] for b in snap["series"]["c1"]] == [T0]
    assert snap["bucket_s"] == 60
    assert snap["to"] > snap["from"]


def test_memory_is_bounded_by_retention_window():
    tl = Timeline(retention_minutes=30)
    for i in range(500):
        tl.add("c1", T0 + i * 60, {"person": 1})
    assert tl.size() <= 31


def test_empty_counts_and_unknown_classes_are_ignored():
    tl = Timeline()
    tl.add("c1", T0, {})
    tl.add("c1", T0, {"person": 0, "dragon": 5})
    assert tl.snapshot(T0, minutes=5)["series"] == {"all": []}


def test_cameras_are_bounded():
    tl = Timeline(max_cameras=2)
    for i in range(5):
        tl.add(f"c{i}", T0, {"person": 1})
    assert len([k for k in tl.snapshot(T0, minutes=5)["series"] if k != "all"]) == 2
