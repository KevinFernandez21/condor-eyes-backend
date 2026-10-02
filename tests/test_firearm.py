"""Tests del prototipo de armas: conversión/filtrado, config, dataset y smoke de video."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from firearm import (
    FirearmConfig,
    FirearmDetector,
    annotate_video,
    boxes_to_dets,
    latency_stats,
    load_config,
)
from firearm.dataset import (
    coco_bbox_to_yolo,
    convert_split,
    scene_of,
    split_yolo_by_scene,
    write_data_yaml,
)

NAMES = {0: "person", 1: "handgun"}


# --- conversión y filtrado de resultados -----------------------------------


def test_boxes_to_dets_converts_and_sorts_by_conf():
    dets = boxes_to_dets(
        [[0, 0, 10, 10], [5, 5, 20, 20]],
        [0.0, 1.0],
        [0.4, 0.9],
        NAMES,
    )
    assert [d["label"] for d in dets] == ["handgun", "person"]
    assert dets[0] == {
        "xyxy": [5.0, 5.0, 20.0, 20.0],
        "cls": 1,
        "conf": 0.9,
        "label": "handgun",
    }


def test_boxes_to_dets_filters_by_conf_and_class():
    xyxy = [[0, 0, 1, 1]] * 3
    dets = boxes_to_dets(
        xyxy, [1, 1, 0], [0.2, 0.6, 0.95], NAMES, min_conf=0.5, keep=["handgun"]
    )
    assert len(dets) == 1 and dets[0]["conf"] == pytest.approx(0.6)


def test_boxes_to_dets_empty_keep_means_all_and_unknown_class_uses_id():
    dets = boxes_to_dets([[0, 0, 1, 1]], [7], [0.5], NAMES, keep=[])
    assert dets[0]["label"] == "7"


def test_boxes_to_dets_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        boxes_to_dets([[0, 0, 1, 1]], [0, 1], [0.5], NAMES)


def test_latency_stats_percentiles():
    s = latency_stats([float(i) for i in range(1, 101)])
    assert s["p50_ms"] == 51.0 and s["p90_ms"] == 91.0
    assert latency_stats([])["p50_ms"] == 0.0


# --- config ----------------------------------------------------------------


def test_load_config_toml_and_overrides(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text('[detector]\nmodel = "m.pt"\nconf = 0.25\n', encoding="utf-8")
    cfg = load_config(p, conf=0.6, iou=None)
    assert cfg.model == "m.pt" and cfg.conf == 0.6 and cfg.iou == FirearmConfig().iou


def test_repo_config_is_valid():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "firearm.toml")
    assert cfg.weapon_classes == ["weapon"] and "weapon" in cfg.classes


@pytest.mark.parametrize("bad", ["conf = 1.5", "imgsz = 600", "foo = 1"])
def test_load_config_rejects_invalid(tmp_path: Path, bad: str):
    p = tmp_path / "c.toml"
    p.write_text(f"[detector]\n{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


# --- detector con modelo simulado (sin GPU ni pesos) ----------------------


class _FakeTensor(list):
    def tolist(self):
        return list(self)


class _FakeYOLO:
    names = NAMES

    def __init__(self):
        self.calls: list[dict] = []

    def predict(self, frame, **kw):
        self.calls.append(kw)
        boxes = SimpleNamespace(
            xyxy=_FakeTensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            cls=_FakeTensor([1.0, 0.0]),
            conf=_FakeTensor([0.8, 0.1]),
        )
        return [SimpleNamespace(boxes=boxes, names=NAMES)]


def test_firearm_detector_infer_uses_config():
    det = FirearmDetector(FirearmConfig(conf=0.3, device="cpu", classes=["handgun"]))
    fake = _FakeYOLO()
    det._model, det._class_ids = fake, [1]
    dets = det.infer(np.zeros((32, 32, 3), dtype="uint8"))
    assert [d["label"] for d in dets] == ["handgun"]
    kw = fake.calls[0]
    assert kw["conf"] == 0.3 and kw["classes"] == [1] and kw["quantize"] is None


# --- dataset COCO -> YOLO ---------------------------------------------------


def test_coco_bbox_to_yolo_normalizes_and_clips():
    assert coco_bbox_to_yolo([0, 0, 50, 100], 100, 200) == (0.25, 0.25, 0.5, 0.5)
    assert coco_bbox_to_yolo([90, 0, 20, 10], 100, 100) == pytest.approx(
        (0.95, 0.05, 0.1, 0.1)
    )
    assert coco_bbox_to_yolo([200, 200, 5, 5], 100, 100) is None


def test_convert_split_writes_labels(tmp_path: Path):
    import cv2

    imgs = tmp_path / "all_images"
    imgs.mkdir()
    cv2.imwrite(str(imgs / "a.jpg"), np.zeros((100, 200, 3), dtype="uint8"))
    coco = {
        "categories": [{"id": 2, "name": "handgun"}, {"id": 1, "name": "person"}],
        "images": [
            {"id": "a", "file_name": "a.jpg", "width": 200, "height": 100},
            {"id": "b", "file_name": "missing.jpg", "width": 10, "height": 10},
        ],
        "annotations": [
            {"image_id": "a", "category_id": 2, "bbox": [0, 0, 20, 10], "iscrowd": 0},
            {"image_id": "a", "category_id": 1, "bbox": [0, 0, 1, 1], "iscrowd": 1},
        ],
    }
    j = tmp_path / "ann.json"
    j.write_text(json.dumps(coco), encoding="utf-8")
    out = tmp_path / "yolo"
    stats = convert_split(j, imgs, out, "train")
    assert stats["names"] == ["person", "handgun"]
    assert stats["images"] == 1 and stats["missing"] == 1 and stats["boxes"] == 1
    assert (out / "labels/train/a.txt").read_text().split()[0] == "1"
    assert (out / "images/train/a.jpg").exists()
    yaml = write_data_yaml(out, stats["names"], {"train": "train"}).read_text()
    assert "1: handgun" in yaml


def _yolo_src(tmp_path: Path) -> Path:
    import cv2

    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir()
    for scene, n in (("Scene1", 3), ("Scene2", 2), ("Scene3", 2)):
        for i in range(1, n + 1):
            cv2.imwrite(
                str(src / "images" / f"{scene}_{i}.png"),
                np.zeros((8, 8, 3), dtype="uint8"),
            )
            (src / "labels" / f"{scene}_{i}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n"
            )
    return src


def test_scene_of():
    assert scene_of("Scene3_12") == "Scene3" and scene_of("frame") == "frame"


def test_split_yolo_by_scene_keeps_scenes_apart(tmp_path: Path):
    out = tmp_path / "out"
    report = split_yolo_by_scene(
        _yolo_src(tmp_path), out, ["person", "weapon"], ["Scene2"], ["Scene3"]
    )
    by = {s["split"]: s for s in report["splits"]}
    assert (by["train"]["images"], by["val"]["images"], by["test"]["images"]) == (
        3,
        2,
        2,
    )
    assert by["val"]["scenes"] == ["Scene2"] and by["test"]["per_class"] == {
        "person": 2,
        "weapon": 2,
    }
    assert sorted(p.name for p in (out / "labels/test").iterdir()) == [
        "Scene3_1.txt",
        "Scene3_2.txt",
    ]
    assert "test: images/test" in (out / "data.yaml").read_text()


@pytest.mark.parametrize(
    ("val", "test"), [(["Scene2"], ["Scene2"]), (["Nope"], ["Scene3"])]
)
def test_split_yolo_by_scene_rejects_bad_scenes(tmp_path: Path, val, test):
    with pytest.raises(ValueError):
        split_yolo_by_scene(
            _yolo_src(tmp_path), tmp_path / "o", ["person", "weapon"], val, test
        )


def test_split_yolo_by_scene_rejects_out_of_range_class(tmp_path: Path):
    src = _yolo_src(tmp_path)
    (src / "labels" / "Scene1_1.txt").write_text("5 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError):
        split_yolo_by_scene(
            src, tmp_path / "o", ["person", "weapon"], ["Scene2"], ["Scene3"]
        )


# --- smoke: video corto de punta a punta ------------------------------------


class _ScriptedDetector:
    """Arma en los frames pares, nada en los impares (escena negativa)."""

    name = "scripted"

    def __init__(self):
        self.i = 0
        self.closed = False

    def warmup(self, n=5):
        pass

    def infer(self, frame):
        self.i += 1
        if self.i % 2:
            return []
        return [{"xyxy": [10, 10, 60, 50], "cls": 1, "conf": 0.87, "label": "handgun"}]

    def close(self):
        self.closed = True


def _make_video(path: Path, n: int = 12, size=(160, 120)) -> None:
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, size)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), 40 + i * 10, dtype="uint8")
        w.write(frame)
    w.release()


def test_annotate_video_smoke(tmp_path: Path):
    import cv2

    src, dst = tmp_path / "in.mp4", tmp_path / "out" / "annotated.mp4"
    _make_video(src)
    det = _ScriptedDetector()
    stats = annotate_video(det, src, dst, warmup=0)

    assert det.closed
    assert stats["frames"] == 12 and stats["batch"] == 1
    assert stats["frames_with_weapon"] == 6
    assert {"p50_ms", "p90_ms", "avg_fps_end_to_end", "peak_rss_mb"} <= stats.keys()
    assert json.loads(dst.with_suffix(".json").read_text())["frames"] == 12

    cap = cv2.VideoCapture(str(dst))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert len(frames) == 12 and frames[0].shape == (120, 160, 3)
    # El frame 2 (con arma) tiene caja roja; el 1 (negativo) no.
    red = lambda f: int(((f[..., 2] > 200) & (f[..., 0] < 60)).sum())
    assert red(frames[1]) > 50 and red(frames[0]) == 0


def test_annotate_video_max_frames_and_missing_input(tmp_path: Path):
    src = tmp_path / "in.mp4"
    _make_video(src)
    stats = annotate_video(_ScriptedDetector(), src, tmp_path / "o.mp4", max_frames=4)
    assert stats["frames"] == 4
    with pytest.raises(FileNotFoundError):
        annotate_video(_ScriptedDetector(), tmp_path / "nope.mp4", tmp_path / "x.mp4")
