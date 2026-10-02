"""Detectores del repo a través del pipeline compartido y del runtime multiagente (#39)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import numpy as np
import pytest

from agents.handlers import build_default_handlers
from agents.runtime import AgentRuntime
from bus.hub import Topic
from bus.memory import InMemoryHub
from pipeline.detection import DetectorProcessor, to_json_detection
from pipeline.live_pipeline import LiveVideoPipeline, PipelineConfig
from pipeline.publisher import MetadataPublisher
from pipeline.shared_pipeline import StreamSource
from pipeline.sources import Frame, SourceError


class NumpyDetector:
    """Detector que devuelve escalares numpy, como hacen muchos backends."""

    name = "fake-numpy"

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def warmup(self, n: int = 5) -> None:
        pass

    def infer(self, frame: Any) -> list[dict]:
        self.calls += 1
        v = float(frame[0, 0, 0])  # la detección depende del contenido del frame
        return [
            {
                "xyxy": np.array([v, 1.0, v + 10, 20.0]),
                "cls": np.int64(1),
                "conf": np.float32(0.75),
                "label": "weapon",
            }
        ]

    def close(self) -> None:
        self.closed = True


def test_to_json_detection_converts_numpy_and_rejects_bad_values():
    d = to_json_detection(
        {
            "xyxy": np.array([1.234, 2, 3, 4]),
            "cls": np.int64(2),
            "conf": np.float32(0.5),
        },
        "m",
    )
    assert d == {"model": "m", "cls": 2, "conf": 0.5, "xyxy": [1.2, 2.0, 3.0, 4.0]}
    assert all(type(v) is float for v in d["xyxy"]) and type(d["cls"]) is int
    json.dumps(d)
    with pytest.raises(ValueError):
        to_json_detection({"xyxy": [0, 0, float("nan"), 1], "cls": 0, "conf": 0.5}, "m")
    with pytest.raises(ValueError):
        to_json_detection({"xyxy": [0, 0, 1], "cls": 0, "conf": 0.5}, "m")


def test_processor_shares_one_detector_instance_and_warms_up_once():
    det = NumpyDetector()
    proc = DetectorProcessor([det], warmup=2)
    img = np.full((8, 8, 3), 5, dtype=np.uint8)
    proc.warmup(img)
    out_a = proc("cam-1", Frame(8, 8, img))
    out_b = proc("cam-2", Frame(8, 8, img))
    assert det.calls == 2 + 2  # 2 de warmup + 1 por frame, sin warmup por cámara
    assert (
        out_a
        == out_b
        == [
            {
                "model": "fake-numpy",
                "cls": 1,
                "conf": 0.75,
                "xyxy": [5.0, 1.0, 15.0, 20.0],
                "label": "weapon",
            }
        ]
    )
    assert proc("cam-1", Frame(8, 8, None)) == []
    proc.close()
    assert det.closed


def test_processor_requires_a_detector():
    with pytest.raises(ValueError):
        DetectorProcessor([])


class _ListSource:
    def __init__(self, images: list[np.ndarray], gate: threading.Event) -> None:
        self.images, self.gate, self.i = images, gate, 0

    def open(self) -> None:
        if self.i >= len(self.images):
            raise SourceError("agotado")

    def read(self) -> Frame:
        if self.i >= len(self.images):
            time.sleep(0.05)
            raise SourceError("agotado")
        self.gate.wait()
        self.gate.clear()
        img = self.images[self.i]
        self.i += 1
        return Frame(width=img.shape[1], height=img.shape[0], data=img)

    def close(self) -> None:
        pass


class _Sink:
    async def save(self, envelope: Any) -> None:
        pass

    async def send(self, envelope: Any) -> None:
        pass


def test_detections_through_pipeline_bus_and_runtime_match_standalone():
    images = [np.full((16, 16, 3), v, dtype=np.uint8) for v in (3, 7, 11, 13)]
    standalone = [
        DetectorProcessor([NumpyDetector()], warmup=0)("s", Frame(16, 16, im))
        for im in images
    ]

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    run = lambda c: asyncio.run_coroutine_threadsafe(c, loop).result(10)

    async def make_hub() -> InMemoryHub:
        return InMemoryHub()

    hub = run(make_hub())
    runtime = AgentRuntime(
        hub, build_default_handlers(storage_sink=_Sink(), alert_sink=_Sink())
    )
    received: list[dict[str, Any]] = []

    async def collect() -> None:
        async for env in hub.subscribe(Topic.DETECTIONS):
            received.append(dict(env.payload))

    task = asyncio.run_coroutine_threadsafe(collect(), loop)
    run(runtime.start())
    gate = threading.Event()
    gate.set()
    src = _ListSource(images, gate)
    det = NumpyDetector()
    pipe = LiveVideoPipeline(
        lambda _s: src,
        config=PipelineConfig(watchdog_timeout=30.0),
        processor=DetectorProcessor([det], warmup=0),
        publisher=MetadataPublisher(hub, loop),
        on_metadata=lambda _m: gate.set(),
    )
    pipe.add_source(StreamSource("cam-1", "usb:0"))
    pipe.start()
    t_end = time.time() + 10
    while len(received) < len(images) and time.time() < t_end:
        time.sleep(0.02)
    pipe.stop()
    run(runtime.wait_idle(timeout=5.0))
    health = runtime.health()
    run(runtime.stop())
    task.cancel()
    run(hub.close())
    loop.call_soon_threadsafe(loop.stop)

    assert [
        r["detections"] for r in sorted(received, key=lambda r: r["frame_index"])
    ] == standalone
    # Solo metadata: ningún array ni buffer en lo que llegó por el bus.
    for r in received:
        json.dumps(r)
        assert set(r) == {
            "stream_id",
            "frame_index",
            "timestamp",
            "width",
            "height",
            "detections",
        }
    tracker = next(v for v in health.values() if v["role"] == "tracker")
    assert tracker["processed"] == len(images) and tracker["failures"] == 0
