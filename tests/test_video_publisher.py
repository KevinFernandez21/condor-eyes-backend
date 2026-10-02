"""El único camino pipeline -> bus es MetadataPublisher, que valida el payload."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime

import numpy as np
import pytest

from bus import MetadataEnvelope, Topic
from pipeline import (
    BackoffPolicy,
    FakeSourceFactory,
    FrameMetadata,
    LiveVideoPipeline,
    MetadataPublisher,
    PipelineConfig,
    StreamHealth,
    StreamSource,
    StreamState,
)


class RecordingHub:
    """MetadataHub mínimo que registra lo publicado."""

    def __init__(self) -> None:
        self.published: list[tuple[Topic, MetadataEnvelope]] = []

    async def publish(self, topic: Topic, message: MetadataEnvelope) -> None:
        self.published.append((topic, message))

    def subscribe(self, topic: Topic):  # pragma: no cover - no se usa
        raise NotImplementedError


@pytest.fixture
def loop():
    event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()
    yield event_loop
    event_loop.call_soon_threadsafe(event_loop.stop)
    thread.join(timeout=2)
    event_loop.close()


def wait_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


def meta(**kwargs) -> FrameMetadata:
    base = {
        "stream_id": "cam-01",
        "frame_index": 1,
        "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "width": 64,
        "height": 48,
    }
    base.update(kwargs)
    return FrameMetadata(**base)


def test_publish_metadata_goes_to_detections_topic(loop):
    hub = RecordingHub()
    publisher = MetadataPublisher(hub, loop)
    publisher.publish_metadata(meta(detections=({"label": "person"},)))
    assert wait_until(lambda: len(hub.published) == 1)
    topic, envelope = hub.published[0]
    assert topic is Topic.DETECTIONS
    assert envelope.source == "pipeline" and envelope.stream_id == "cam-01"
    assert envelope.payload["detections"] == [{"label": "person"}]


def test_publish_status_goes_to_stream_status_topic(loop):
    hub = RecordingHub()
    publisher = MetadataPublisher(hub, loop)
    publisher.publish_status(
        StreamHealth("cam-01", StreamState.RECONNECTING, last_error="boom")
    )
    assert wait_until(lambda: len(hub.published) == 1)
    topic, envelope = hub.published[0]
    assert topic is Topic.STREAM_STATUS
    assert envelope.payload["state"] == "reconnecting"


def test_publisher_refuses_pixels_even_if_metadata_was_tampered(loop):
    hub = RecordingHub()
    publisher = MetadataPublisher(hub, loop)
    bad = meta()
    object.__setattr__(bad, "detections", ({"crop": np.zeros((2, 2, 3))},))
    with pytest.raises(TypeError):
        publisher.publish_metadata(bad)
    with pytest.raises(TypeError):
        publisher.publish_payload(Topic.EVENTS, "x", {"jpeg": b"\xff\xd8"})
    time.sleep(0.05)
    assert hub.published == []


def test_publisher_bounds_in_flight_messages(loop):
    class StuckHub(RecordingHub):
        async def publish(self, topic, message):
            await asyncio.sleep(0.3)

    publisher = MetadataPublisher(StuckHub(), loop, max_pending=3)
    for index in range(20):
        publisher.publish_metadata(meta(frame_index=index))
    assert publisher.pending <= 3
    assert publisher.dropped >= 17
    assert wait_until(lambda: publisher.pending == 0)


def test_pipeline_publishes_through_the_publisher_only(loop):
    hub = RecordingHub()
    publisher = MetadataPublisher(hub, loop)
    factory = FakeSourceFactory(default_interval=0.002)
    pipeline = LiveVideoPipeline(
        factory,
        config=PipelineConfig(
            backoff=BackoffPolicy(initial=0.01, maximum=0.05),
            watchdog_timeout=1.0,
            watchdog_interval=0.05,
        ),
        publisher=publisher,
    )
    pipeline.add_source(StreamSource("cam-a", "rtsp://fake/a"))
    pipeline.start()
    try:
        assert wait_until(
            lambda: sum(t is Topic.DETECTIONS for t, _ in hub.published) >= 3
        )
        assert wait_until(
            lambda: any(t is Topic.STREAM_STATUS for t, _ in hub.published)
        )
    finally:
        pipeline.stop()
    for _, envelope in hub.published:
        assert not any(
            isinstance(v, (bytes, bytearray, memoryview, np.ndarray))
            for v in envelope.payload.values()
        )


def test_pipeline_blocks_leaking_processor_before_the_bus(loop):
    hub = RecordingHub()
    publisher = MetadataPublisher(hub, loop)
    pipeline = LiveVideoPipeline(
        FakeSourceFactory(default_interval=0.002),
        config=PipelineConfig(
            backoff=BackoffPolicy(initial=0.01, maximum=0.05),
            watchdog_timeout=1.0,
            watchdog_interval=0.05,
        ),
        publisher=publisher,
        processor=lambda stream_id, frame: [{"frame": frame.data, "arr": np.zeros(3)}],
    )
    pipeline.add_source(StreamSource("cam-a", "rtsp://fake/a"))
    pipeline.start()
    try:
        assert wait_until(lambda: pipeline.stream_health("cam-a").callback_errors >= 2)
    finally:
        pipeline.stop()
    assert not any(t is Topic.DETECTIONS for t, _ in hub.published)
