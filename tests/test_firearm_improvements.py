"""Tests del evaluador común, el detector en dos etapas y los aumentos de dominio."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import cv2
import numpy as np
import pytest

from firearm.augment import degrade, make_composites, make_degraded
from firearm.evaluate import calibrate, load_split, metrics, negatives_rate
from firearm.twostage import TwoStageConfig, TwoStageDetector, _nms

# --- evaluador ----------------------------------------------------------------


def test_metrics_ap_precision_recall():
    gts = [[[0, 0, 10, 10]], [[20, 20, 40, 40]], []]
    preds = [
        [(0.9, [0, 0, 10, 10])],
        [(0.8, [20, 20, 40, 40]), (0.3, [0, 0, 5, 5])],
        [(0.6, [1, 1, 2, 2])],
    ]
    m = metrics(preds, gts, thr=0.5)
    assert m["ap50"] == pytest.approx(1.0)
    assert (m["tp"], m["fp"], m["gt"]) == (2, 1, 2) and m["recall"] == 1.0
    assert metrics(preds, gts, thr=0.7)["precision"] == 1.0
    assert calibrate(preds, gts) > 0.6


def test_negatives_rate():
    r = negatives_rate([[(0.9, [0, 0, 1, 1])], [(0.2, [0, 0, 1, 1])], []], thr=0.5)
    assert r == {"images": 3, "with_false_alarm": 1, "rate": pytest.approx(1 / 3)}


def _yolo(root: Path, split: str, stem: str, labels: str, size=(200, 100)) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    img = (np.random.default_rng(0).random((size[1], size[0], 3)) * 255).astype("uint8")
    cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), img)
    (root / "labels" / split / f"{stem}.txt").write_text(labels)


def test_load_split_keeps_only_weapon_boxes_in_pixels(tmp_path: Path):
    _yolo(tmp_path, "test", "a", "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.1 0.2\n")
    items = load_split(tmp_path, "test")
    assert items[0][1] == [
        [
            pytest.approx(90.0),
            pytest.approx(40.0),
            pytest.approx(110.0),
            pytest.approx(60.0),
        ]
    ]


# --- dos etapas ------------------------------------------------------------------


class _T(list):
    def tolist(self):
        return list(self)


def _res(boxes, confs):
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=_T(boxes), conf=_T(confs)))


def test_nms_keeps_best_and_distinct():
    out = _nms(
        [(0.5, [0, 0, 10, 10]), (0.9, [1, 1, 10, 10]), (0.7, [50, 50, 60, 60])], 0.5
    )
    assert [c for c, _ in out] == [0.9, 0.7]


def test_two_stage_maps_crop_detections_back_to_frame():
    det = TwoStageDetector(
        TwoStageConfig(
            device="cpu", crop_size=100, pad_x=0.0, pad_y=0.0, full_frame=False
        )
    )
    frame = np.zeros((400, 400, 3), dtype="uint8")
    calls = []

    class Person:
        def predict(self, img, **kw):
            return [_res([[100, 100, 150, 300]], [0.9])]  # persona de 50×200 px

    class Weapon:
        names: ClassVar[dict[int, str]] = {0: "person", 1: "weapon"}

        def predict(self, imgs, **kw):
            calls.append([i.shape for i in imgs])
            # Arma en el recorte reescalado (200 px → 100 px: escala 0,5).
            return [_res([[10, 40, 20, 60]], [0.8])]

    det._person, det._weapon, det._weapon_cls = Person(), Weapon(), 1
    dets = det.infer(frame)
    weapon = next(d for d in dets if d["label"] == "weapon")
    assert weapon["xyxy"] == pytest.approx([120, 180, 140, 220])
    assert calls[0] == [(100, 25, 3)]
    assert any(d["label"] == "person" for d in dets)


# --- aumentos -----------------------------------------------------------------------


def test_degrade_keeps_shape_and_changes_pixels():
    img = (np.random.default_rng(1).random((120, 160, 3)) * 255).astype("uint8")
    out = degrade(img, random.Random(0))
    assert (
        out.shape == img.shape
        and out.dtype == np.uint8
        and not np.array_equal(out, img)
    )


def test_make_degraded_copies_labels(tmp_path: Path):
    src = tmp_path / "src"
    for i in range(4):
        _yolo(src, "train", f"x{i}", "1 0.5 0.5 0.1 0.1\n")
    n = make_degraded(src, "train", tmp_path / "out", frac=0.5)
    labs = sorted((tmp_path / "out" / "labels" / "train").glob("deg_*.txt"))
    assert n == 2 and len(labs) == 2 and labs[0].read_text().startswith("1 ")


def test_make_composites_adds_small_armed_person(tmp_path: Path):
    src, bg = tmp_path / "oi", tmp_path / "bg"
    # Persona alta con un arma dentro; fondo de 400×300 con una persona propia.
    _yolo(src, "train", "a", "0 0.5 0.5 0.4 0.8\n1 0.55 0.5 0.2 0.1\n", size=(400, 400))
    _yolo(bg, "train", "b", "0 0.1 0.5 0.05 0.2\n", size=(400, 300))
    r = make_composites(src, [(bg, "train")], tmp_path / "out", n=3, height_px=(60, 80))
    assert r["composites"] == 3 and r["armed_people_sources"] == 1
    for lab in (tmp_path / "out" / "labels" / "train").glob("cp_*.txt"):
        rows = [ln.split() for ln in lab.read_text().splitlines()]
        assert sum(r[0] == "1" for r in rows) == 1
        h = max(float(r[4]) for r in rows if r[0] == "0") * 300
        assert h <= 80 * 1.05  # persona pegada pequeña (o la del fondo)
