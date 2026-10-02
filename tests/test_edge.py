"""Tests del presupuesto edge: letterbox, posproceso del detector lean y matriz de configuraciones."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from edge.budget import _pct
from edge.ovdetect import OVDetector, letterbox

ROOT = Path(__file__).resolve().parents[1]


def test_letterbox_keeps_aspect_and_pads():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out, r, (px, py) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert r == pytest.approx(1 / 3) and px == 0 and py == 140
    assert out[0, 0, 0] == 114 and out[320, 320, 0] == 0


def test_lean_detector_postprocess_maps_and_suppresses():
    det = OVDetector.__new__(OVDetector)
    det.imgsz, det.conf, det.iou = 640, 0.3, 0.6
    nc = 2
    anchors = np.zeros((4 + nc, 3), dtype=np.float32)
    # Dos cajas casi iguales de la clase 1 (una se suprime) y una caja de baja confianza.
    anchors[:, 0] = [320, 330, 90, 90, 0.0, 0.9]
    anchors[:, 1] = [322, 330, 90, 90, 0.0, 0.7]
    anchors[:, 2] = [100, 200, 20, 20, 0.1, 0.1]
    det.compiled = lambda inputs: [np.stack([anchors] * len(inputs[0]))]
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res = det.predict([frame, frame])
    assert len(res) == 2 and len(res[0]) == 1
    cls, conf, (x1, y1, x2, y2) = res[0][0]
    assert cls == 1 and conf == pytest.approx(0.9)
    # Centro (320, 330) en la entrada 640 → frame 1080p: escala 3, relleno vertical 140.
    assert ((x1 + x2) / 2, (y1 + y2) / 2) == (pytest.approx(960), pytest.approx(570))
    assert x2 - x1 == pytest.approx(270)


def test_percentile():
    assert _pct([float(i) for i in range(1, 11)], 0.5) == 6.0
    assert _pct([], 0.9) == 0.0


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "edge_budget_script", ROOT / "scripts" / "edge_budget.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_matrix_covers_required_cases():
    cfgs = _load_script().matrix()
    assert {c["device"] for c in cfgs} == {"intel:gpu.0", "intel:cpu"}
    assert {c["streams"] for c in cfgs} >= {1, 4, 8}
    assert {d["imgsz"] for c in cfgs for d in c["detectors"]} == {640, 1280}
    assert any(c["source"] == "720p" for c in cfgs)
    assert any(c.get("reid") and c.get("face") for c in cfgs)
    # Nunca un modelo por cámara: el número de detectores no depende de N.
    for c in cfgs:
        assert len(c["detectors"]) <= 2
