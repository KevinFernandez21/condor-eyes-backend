"""Backend CPU (OpenCV, con cv2 simulado) y descripción de grafos GStreamer."""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from pipeline import OpenCVFrameSource, SourceError, StreamSource
from pipeline.gst_graph import cpu_graph, deepstream_graph


class FakeCapture:
    opened_with: ClassVar[list] = []

    def __init__(self, target, frames: int = 2, ok: bool = True) -> None:
        FakeCapture.opened_with.append(target)
        self._ok = ok
        self._frames = frames
        self.released = False

    def isOpened(self) -> bool:
        return self._ok

    def read(self):
        if self._frames <= 0:
            return False, None
        self._frames -= 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


@pytest.fixture
def fake_cv2(monkeypatch):
    FakeCapture.opened_with = []
    module = types.SimpleNamespace(VideoCapture=FakeCapture)
    monkeypatch.setitem(sys.modules, "cv2", module)
    return module


def test_opencv_source_reads_frames_and_reports_size(fake_cv2):
    source = OpenCVFrameSource(StreamSource("cam", "rtsp://host/stream"))
    source.open()
    frame = source.read()
    assert (frame.width, frame.height) == (64, 48)
    source.close()
    source.close()  # idempotente
    assert FakeCapture.opened_with == ["rtsp://host/stream"]


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("usb:0", 0),
        ("usb://2", 2),
        ("/dev/video1", "/dev/video1"),
        ("v4l2:///dev/video3", "/dev/video3"),
    ],
)
def test_opencv_usb_targets(fake_cv2, uri, expected):
    OpenCVFrameSource(StreamSource("cam", uri)).open()
    assert FakeCapture.opened_with == [expected]


def test_opencv_source_raises_in_spanish_when_open_or_read_fails(fake_cv2, monkeypatch):
    monkeypatch.setattr(
        fake_cv2, "VideoCapture", lambda target: FakeCapture(target, ok=False)
    )
    with pytest.raises(SourceError, match="No se pudo abrir"):
        OpenCVFrameSource(StreamSource("cam", "rtsp://host/x")).open()

    monkeypatch.setattr(
        fake_cv2, "VideoCapture", lambda target: FakeCapture(target, frames=0)
    )
    source = OpenCVFrameSource(StreamSource("cam", "rtsp://host/x"))
    source.open()
    with pytest.raises(SourceError, match="Sin frames"):
        source.read()

    with pytest.raises(SourceError, match="no está abierta"):
        OpenCVFrameSource(StreamSource("cam", "rtsp://host/x")).read()


def test_importing_pipeline_does_not_import_heavy_dependencies():
    import subprocess

    code = (
        "import sys; sys.path.insert(0, 'src'); import pipeline; "
        "assert all(m not in sys.modules for m in ('cv2', 'gi', 'tensorrt')), "
        "[m for m in ('cv2','gi','tensorrt') if m in sys.modules]"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_cpu_graph_rtsp_has_bounded_dropping_appsink():
    graph = cpu_graph(StreamSource("cam", "rtsp://10.0.0.5/s1"))
    assert "rtspsrc location=rtsp://10.0.0.5/s1" in graph
    assert "drop-on-latency=true" in graph
    assert "appsink" in graph and "max-buffers=" in graph and "drop=true" in graph
    assert not any(el in graph for el in ("nvv4l2decoder", "nvstreammux", "nvinfer"))


def test_cpu_graph_usb_uses_v4l2():
    graph = cpu_graph(StreamSource("cam", "usb:1"))
    assert "v4l2src device=/dev/video1" in graph


def test_deepstream_graph_matches_target_architecture():
    sources = [
        StreamSource("cam-1", "rtsp://10.0.0.5/s1"),
        StreamSource("cam-2", "/dev/video0"),
    ]
    graph = deepstream_graph(
        sources, engine_config="configs/pgie_yolov8n_fp16.txt", width=1920, height=1080
    )
    order = ["nvv4l2decoder", "nvstreammux", "nvinfer", "nvtracker", "appsink"]
    positions = [graph.index(element) for element in order]
    assert positions == sorted(positions)
    assert "batch-size=2" in graph
    assert "config-file-path=configs/pgie_yolov8n_fp16.txt" in graph
    assert "sink_0" in graph and "sink_1" in graph


def test_deepstream_graph_rejects_invalid_batches():
    with pytest.raises(ValueError, match="batch"):
        deepstream_graph([], engine_config="x.txt")
    too_many = [StreamSource(f"c{i}", f"rtsp://h/{i}") for i in range(9)]
    with pytest.raises(ValueError, match="batch"):
        deepstream_graph(too_many, engine_config="x.txt")
