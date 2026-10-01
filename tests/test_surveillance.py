"""Tests del detector de vigilancia: métricas, datasets y contrato del detector."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from surveillance import (
    NAMES,
    SurveillanceConfig,
    SurveillanceDetector,
    load_surveillance_config,
)
from surveillance.classes import COCO80_TO_IDX, COCO_ID_TO_IDX
from surveillance.dataset import coco_split, mot_split, write_yaml, yolo_person_only
from surveillance.metrics import (
    average_precision,
    best_f1_threshold,
    evaluate,
    iou_matrix,
    load_yolo_gt,
    match_class,
    negative_fp_rate,
    person_recall_buckets,
    pr_at,
)

ROOT = Path(__file__).resolve().parents[1]


def test_class_maps_are_consistent():
    assert NAMES[0] == "person" and len(NAMES) == 9
    assert COCO_ID_TO_IDX[1] == 0 and COCO80_TO_IDX[0] == 0
    assert COCO80_TO_IDX[28] == NAMES.index("suitcase")


# --- métricas -----------------------------------------------------------------


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    assert iou_matrix(a, b).round(3).tolist() == [[1.0, 0.333, 0.0]]


def test_perfect_predictions_give_ap_one():
    gts = {"a": [(0, [0, 0, 10, 10])], "b": [(0, [5, 5, 50, 50])]}
    preds = {"a": [(0, 0.9, [0, 0, 10, 10])], "b": [(0, 0.8, [5, 5, 50, 50])]}
    m = match_class(preds, gts, 0)
    assert m.n_gt == 2 and m.tp.all()
    assert average_precision(m) == pytest.approx(np.ones(10))


def test_duplicates_are_false_positives_and_thresholds_work():
    gts = {"a": [(0, [0, 0, 10, 10])]}
    preds = {
        "a": [
            (0, 0.9, [0, 0, 10, 10]),
            (0, 0.4, [0, 0, 10, 10]),
            (0, 0.2, [50, 50, 60, 60]),
        ]
    }
    m = match_class(preds, gts, 0)
    assert pr_at(m, 0.3) == (0.5, 1.0, 1, 1)
    assert pr_at(m, 0.5) == (1.0, 1.0, 1, 0)
    thr = best_f1_threshold(m)
    assert pr_at(m, thr + 1e-6)[:2] == (1.0, 1.0)


def test_evaluate_and_negative_rate():
    gts = {"a": [(0, [0, 0, 10, 10]), (2, [20, 20, 40, 40])]}
    preds = {"a": [(0, 0.9, [0, 0, 10, 10]), (2, 0.2, [20, 20, 40, 40])]}
    r = evaluate(preds, gts, {"person": 0.5, "car": 0.3})
    assert r["per_class"]["person"]["recall"] == 1.0
    assert r["per_class"]["car"]["recall"] == 0.0 and r["per_class"]["car"][
        "ap50"
    ] == pytest.approx(1.0)
    neg = negative_fp_rate(
        {"x": [(0, 0.6, [0, 0, 1, 1])], "y": [(0, 0.1, [0, 0, 1, 1])]}, {"person": 0.5}
    )
    assert neg["images_with_fp"] == 1 and neg["fp_image_rate"] == 0.5


def test_person_recall_buckets():
    meta = {
        "f": [
            {"xyxy": [0, 0, 10, 40], "visibility": 0.9, "height": 40},
            {"xyxy": [100, 0, 110, 300], "visibility": 0.1, "height": 300},
        ]
    }
    preds = {"f": [(0, 0.9, [0, 0, 10, 40])]}
    b = person_recall_buckets(preds, meta, 0.5)
    assert b["visibility"]["0.75-1.01"]["recall"] == 1.0
    assert b["visibility"]["0.0-0.25"]["recall"] == 0.0
    assert b["height_px"]["0-50"]["gt"] == 1


# --- datasets --------------------------------------------------------------------


def _img(path: Path, w=100, h=50):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype="uint8"))


def test_coco_split_parity_classes_and_negatives(tmp_path: Path):
    for i in (1, 2, 3, 4):
        _img(tmp_path / "imgs" / f"{i:012d}.jpg")
    coco = {
        "images": [
            {
                "id": i,
                "file_name": f"{i:012d}.jpg",
                "width": 100,
                "height": 50,
                "coco_url": "",
            }
            for i in (1, 2, 3, 4)
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 25], "iscrowd": 0},
            {
                "image_id": 1,
                "category_id": 18,
                "bbox": [0, 0, 5, 5],
                "iscrowd": 0,
            },  # perro: fuera de alcance
            {"image_id": 3, "category_id": 3, "bbox": [10, 10, 20, 20], "iscrowd": 0},
        ],
        "categories": [],
    }
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(coco))
    out = tmp_path / "out"
    st = coco_split(
        ann,
        tmp_path / "imgs",
        out,
        "test",
        image_filter="odd",
        negatives=5,
        negatives_split="neg",
    )
    assert st["images"] == 2 and st["boxes"] == 2 and st["negatives"] == 0
    assert (out / "labels/test/000000000001.txt").read_text().split()[0] == "0"
    assert (out / "labels/test/000000000003.txt").read_text().split()[0] == str(
        NAMES.index("car")
    )
    st = coco_split(
        ann,
        tmp_path / "imgs",
        out,
        "x",
        image_filter="even",
        negatives=1,
        negatives_split="neg",
    )
    assert st["images"] == 0 and st["negatives"] == 1
    assert (out / "labels/neg/000000000002.txt").read_text() == ""


def test_mot_split_filters_and_writes_meta(tmp_path: Path):
    seq = tmp_path / "MOT16-09"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nimWidth=100\nimHeight=50\nseqLength=2\n"
    )
    for f in (1, 2):
        _img(seq / "img1" / f"{f:06d}.jpg")
    (seq / "gt" / "gt.txt").write_text(
        "1,1,10,10,10,20,1,1,0.8\n1,2,50,10,10,20,0,1,1.0\n1,3,60,10,10,20,1,4,1.0\n2,1,12,10,10,20,1,7,0.1\n"
    )
    st = mot_split([seq], tmp_path / "out", "test_mot")
    assert (
        st["images"] == 2 and st["boxes"] == 2
    )  # sin `consider`=0 ni clase 4 (bicicleta)
    meta = json.loads((tmp_path / "out/meta/test_mot.json").read_text())
    assert meta["MOT16-09_000002"][0]["visibility"] == 0.1
    st = mot_split([seq], tmp_path / "out2", "train_mot", min_visibility=0.25)
    assert st["boxes"] == 1


def test_yolo_person_only_and_yaml(tmp_path: Path):
    src = tmp_path / "src"
    _img(src / "images" / "a.png")
    (src / "labels").mkdir()
    (src / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n")
    st = yolo_person_only(src, tmp_path / "out", "test_simuletic")
    assert st["boxes"] == 1
    y = write_yaml(
        tmp_path / "out", "train", {"train": ["a", "b"], "val": ["c"]}
    ).read_text()
    assert "  - images/a" in y and "8: suitcase" in y


def test_load_yolo_gt(tmp_path: Path):
    (tmp_path / "x.txt").write_text("2 0.5 0.5 0.5 0.5\n")
    gt = load_yolo_gt(str(tmp_path), {"x": (100, 50), "y": (10, 10)})
    assert gt["x"] == [(2, [25.0, 12.5, 75.0, 37.5])] and gt["y"] == []


# --- detector ----------------------------------------------------------------------


def test_repo_config_and_overrides(tmp_path: Path):
    cfg = load_surveillance_config(ROOT / "configs" / "surveillance.toml")
    assert cfg.model.endswith(".pt") and cfg.threshold("person") == cfg.default_conf
    p = tmp_path / "c.toml"
    p.write_text('[detector]\nversion = "v1"\n[conf]\nperson = 0.4\n')
    assert load_surveillance_config(p).threshold("person") == 0.4
    p.write_text("[conf]\ndog = 0.4\n")
    with pytest.raises(ValueError):
        load_surveillance_config(p)


class _T(list):
    def tolist(self):
        return list(self)


def test_detector_maps_coco80_and_applies_per_class_threshold():
    det = SurveillanceDetector(
        SurveillanceConfig(device="cpu", conf={"person": 0.5, "car": 0.2})
    )
    calls = []

    class Fake:
        def predict(self, frames, **kw):
            calls.append(kw)
            boxes = SimpleNamespace(
                xyxy=_T([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]]),
                cls=_T([0.0, 2.0, 0.0]),
                conf=_T([0.9, 0.25, 0.3]),
            )
            return [SimpleNamespace(boxes=boxes)]

    det._model, det._map = Fake(), dict(COCO80_TO_IDX)
    dets = det.infer(np.zeros((8, 8, 3), dtype="uint8"))
    assert [(d["label"], d["conf"]) for d in dets] == [("person", 0.9), ("car", 0.25)]
    assert calls[0]["conf"] == 0.2 and calls[0]["quantize"] is None
    assert set(calls[0]["classes"]) == set(COCO80_TO_IDX)
