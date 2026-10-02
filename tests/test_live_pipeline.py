"""Ciclo de vida, reconexión, watchdog y backpressure del pipeline en vivo.

Usa fuentes falsas (sin cámaras, GPU ni GStreamer) y backoff de milisegundos.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from bus import MetadataEnvelope
from pipeline import (
    BackoffPolicy,
    FrameMetadata,
    SharedVideoPipeline,
    StreamHealth,
    StreamSource,
    StreamState,
    ensure_metadata_only,
)
from pipeline.fake import FakeCamera, FakeSourceFactory
from pipeline.live_pipeline import LiveVideoPipeline, PipelineConfig


def wait_until(condition: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


def fast_config(**overrides) -> PipelineConfig:
    base = {
        "backoff": BackoffPolicy(initial=0.01, factor=2.0, maximum=0.05),
        "watchdog_timeout": 0.15,
        "watchdog_interval": 0.02,
        "max_buffered_frames": 4,
        "join_timeout": 2.0,
    }
    base.update(overrides)
    return PipelineConfig(**base)


@pytest.fixture
def factory() -> FakeSourceFactory:
    return FakeSourceFactory(default_interval=0.002)


@pytest.fixture
def received() -> list[FrameMetadata]:
    return []


@pytest.fixture
def pipeline(factory, received):
    instance = LiveVideoPipeline(
        factory, config=fast_config(), on_metadata=received.append
    )
    yield instance
    instance.stop()


def src(stream_id: str) -> StreamSource:
    return StreamSource(stream_id, f"rtsp://fake/{stream_id}")


def state_of(pipeline: LiveVideoPipeline, stream_id: str) -> StreamState:
    return pipeline.stream_health(stream_id).state


def test_pipeline_satisfies_the_shared_pipeline_boundary(pipeline):
    boundary: SharedVideoPipeline = pipeline
    assert boundary.health()["running"] is False


def test_lifecycle_start_streams_frames_and_stop(pipeline, received):
    pipeline.add_source(src("cam-01"))
    assert state_of(pipeline, "cam-01") is StreamState.IDLE

    pipeline.start()
    assert wait_until(lambda: len(received) >= 3)
    health = pipeline.stream_health("cam-01")
    assert health.state is StreamState.CONNECTED
    assert health.last_frame_at is not None
    assert health.retry_count == 0
    assert health.last_error is None
    assert pipeline.health()["running"] is True

    pipeline.stop()
    assert state_of(pipeline, "cam-01") is StreamState.STOPPED
    assert pipeline.health()["running"] is False


def test_metadata_has_stream_id_and_increasing_frame_index(pipeline, received):
    pipeline.add_source(src("cam-01"))
    pipeline.start()
    assert wait_until(lambda: len(received) >= 5)
    indices = [m.frame_index for m in received if m.stream_id == "cam-01"]
    assert indices == sorted(indices)
    assert received[0].width > 0 and received[0].height > 0


def test_sources_can_be_added_while_running(pipeline, received):
    pipeline.start()
    pipeline.add_source(src("cam-late"))
    assert wait_until(lambda: any(m.stream_id == "cam-late" for m in received))


def test_duplicate_stream_ids_are_rejected(pipeline):
    pipeline.add_source(src("cam-01"))
    with pytest.raises(ValueError, match="cam-01"):
        pipeline.add_source(StreamSource("cam-01", "rtsp://fake/other"))
    assert len(pipeline.health()["streams"]) == 1


def test_max_streams_is_enforced():
    pipeline = LiveVideoPipeline(FakeSourceFactory(), config=fast_config(max_streams=2))
    pipeline.add_source(src("a"))
    pipeline.add_source(src("b"))
    with pytest.raises(ValueError, match="máximo"):
        pipeline.add_source(src("c"))


def test_remove_source_stops_only_that_stream(pipeline, received):
    pipeline.add_source(src("cam-a"))
    pipeline.add_source(src("cam-b"))
    pipeline.start()
    assert wait_until(lambda: len(received) >= 4)

    pipeline.remove_source("cam-a")
    assert "cam-a" not in pipeline.health()["streams"]
    mark = len(received)
    assert wait_until(lambda: len(received) >= mark + 3)
    assert state_of(pipeline, "cam-b") is StreamState.CONNECTED


def test_remove_unknown_source_raises(pipeline):
    with pytest.raises(KeyError, match="nope"):
        pipeline.remove_source("nope")
    with pytest.raises(KeyError, match="nope"):
        pipeline.stream_health("nope")


def test_failed_stream_reconnects_without_stopping_healthy_ones(
    pipeline, factory, received
):
    cam_a: FakeCamera = factory.camera("cam-a")
    pipeline.add_source(src("cam-a"))
    pipeline.add_source(src("cam-b"))
    pipeline.start()
    assert wait_until(lambda: state_of(pipeline, "cam-a") is StreamState.CONNECTED)

    cam_a.disconnect()
    assert wait_until(lambda: state_of(pipeline, "cam-a") is StreamState.RECONNECTING)
    assert wait_until(lambda: pipeline.stream_health("cam-a").retry_count >= 2)

    health_a = pipeline.stream_health("cam-a")
    assert health_a.last_error is not None
    assert "ConnectionError" in health_a.last_error

    # cam-b sigue entregando frames mientras cam-a está caída
    mark = sum(1 for m in received if m.stream_id == "cam-b")
    assert wait_until(
        lambda: sum(1 for m in received if m.stream_id == "cam-b") >= mark + 5
    )
    assert state_of(pipeline, "cam-b") is StreamState.CONNECTED
    assert pipeline.stream_health("cam-b").retry_count == 0

    cam_a.reconnect()
    assert wait_until(lambda: state_of(pipeline, "cam-a") is StreamState.CONNECTED)
    assert wait_until(lambda: pipeline.stream_health("cam-a").retry_count == 0)
    assert pipeline.stream_health("cam-a").total_reconnects >= 2


def test_open_failures_are_retried_with_backoff(pipeline, factory):
    camera = factory.camera("cam-a")
    camera.fail_opens(3)
    pipeline.add_source(src("cam-a"))
    pipeline.start()
    assert wait_until(lambda: state_of(pipeline, "cam-a") is StreamState.CONNECTED)
    assert camera.open_attempts == 4
    assert pipeline.stream_health("cam-a").total_reconnects == 3


def test_stream_fails_permanently_after_max_retries(factory):
    camera = factory.camera("cam-a")
    camera.disconnect()
    pipeline = LiveVideoPipeline(factory, config=fast_config(max_retries=2))
    pipeline.add_source(src("cam-a"))
    pipeline.add_source(src("cam-b"))
    pipeline.start()
    try:
        assert wait_until(lambda: state_of(pipeline, "cam-a") is StreamState.FAILED)
        assert pipeline.stream_health("cam-a").last_error
        assert state_of(pipeline, "cam-b") is StreamState.CONNECTED
    finally:
        pipeline.stop()


def test_watchdog_recovers_a_stalled_stream(pipeline, factory):
    camera = factory.camera("cam-a")
    pipeline.add_source(src("cam-a"))
    pipeline.add_source(src("cam-b"))
    pipeline.start()
    assert wait_until(lambda: state_of(pipeline, "cam-a") is StreamState.CONNECTED)

    camera.stall()
    assert wait_until(lambda: pipeline.stream_health("cam-a").total_reconnects >= 1)
    assert "watchdog" in (pipeline.stream_health("cam-a").last_error or "")
    assert state_of(pipeline, "cam-b") is StreamState.CONNECTED

    camera.resume()
    assert wait_until(lambda: state_of(pipeline, "cam-a") is StreamState.CONNECTED)
    before = pipeline.stream_health("cam-a").frames_received
    assert wait_until(lambda: pipeline.stream_health("cam-a").frames_received > before)


def test_slow_consumer_keeps_backpressure_bounded(factory):
    gate = threading.Event()
    seen: list[FrameMetadata] = []

    def slow_consumer(meta: FrameMetadata) -> None:
        gate.wait(timeout=5)
        seen.append(meta)

    pipeline = LiveVideoPipeline(
        factory,
        config=fast_config(max_buffered_frames=3),
        on_metadata=slow_consumer,
    )
    pipeline.add_source(src("cam-a"))
    pipeline.start()
    try:
        assert wait_until(lambda: pipeline.stream_health("cam-a").frames_dropped > 10)
        health = pipeline.stream_health("cam-a")
        assert health.queue_depth <= 3
        assert state_of(pipeline, "cam-a") is StreamState.CONNECTED
        # El lector no se bloquea: sigue recibiendo frames aunque el consumidor esté parado.
        before = health.frames_received
        assert wait_until(
            lambda: pipeline.stream_health("cam-a").frames_received > before
        )
    finally:
        gate.set()
        pipeline.stop()


def test_disconnection_does_not_grow_buffers(pipeline, factory):
    camera = factory.camera("cam-a")
    pipeline.add_source(src("cam-a"))
    pipeline.start()
    assert wait_until(lambda: state_of(pipeline, "cam-a") is StreamState.CONNECTED)
    camera.disconnect()
    assert wait_until(lambda: state_of(pipeline, "cam-a") is StreamState.RECONNECTING)
    time.sleep(0.1)
    assert pipeline.stream_health("cam-a").queue_depth <= 4


def test_callback_errors_are_counted_and_do_not_kill_the_stream(factory):
    calls = []

    def flaky(meta: FrameMetadata) -> None:
        calls.append(meta)
        raise RuntimeError("consumidor roto")

    pipeline = LiveVideoPipeline(factory, config=fast_config(), on_metadata=flaky)
    pipeline.add_source(src("cam-a"))
    pipeline.start()
    try:
        assert wait_until(lambda: len(calls) >= 3)
        assert pipeline.stream_health("cam-a").callback_errors >= 3
        assert state_of(pipeline, "cam-a") is StreamState.CONNECTED
    finally:
        pipeline.stop()


def test_stop_is_clean_even_with_stalled_and_reconnecting_streams(factory):
    pipeline = LiveVideoPipeline(
        factory,
        config=fast_config(
            backoff=BackoffPolicy(initial=5.0, factor=2.0, maximum=10.0),
            watchdog_timeout=60.0,
        ),
    )
    factory.camera("cam-down").disconnect()
    factory.camera("cam-stuck").stall()
    pipeline.add_source(src("cam-ok"))
    pipeline.add_source(src("cam-down"))
    pipeline.add_source(src("cam-stuck"))
    baseline = threading.active_count()
    pipeline.start()
    assert wait_until(
        lambda: state_of(pipeline, "cam-down") is StreamState.RECONNECTING
    )

    started = time.monotonic()
    pipeline.stop()
    assert time.monotonic() - started < 2.5
    assert all(
        h["state"] == StreamState.STOPPED.value
        for h in pipeline.health()["streams"].values()
    )
    factory.camera("cam-stuck").resume()
    assert wait_until(lambda: threading.active_count() <= baseline)
    assert all(camera.open_sources == 0 for camera in factory.cameras())


def test_stop_and_start_are_idempotent_and_restartable(pipeline, received):
    pipeline.stop()  # sin start previo
    pipeline.add_source(src("cam-01"))
    pipeline.start()
    pipeline.start()
    assert wait_until(lambda: len(received) >= 2)
    pipeline.stop()
    pipeline.stop()
    mark = len(received)
    pipeline.start()
    assert wait_until(lambda: len(received) >= mark + 2)


def test_processor_output_flows_as_metadata_only(factory):
    received: list[FrameMetadata] = []

    def processor(stream_id, frame):
        return [{"label": "person", "conf": 0.8, "bbox": [0, 0, 10, 10]}]

    pipeline = LiveVideoPipeline(
        factory, config=fast_config(), on_metadata=received.append, processor=processor
    )
    pipeline.add_source(src("cam-a"))
    pipeline.start()
    try:
        assert wait_until(lambda: len(received) >= 2)
        assert received[0].detections[0]["label"] == "person"
    finally:
        pipeline.stop()


def test_processor_leaking_pixels_is_rejected_and_counted(factory):
    received: list[FrameMetadata] = []

    def leaking(stream_id, frame):
        return [{"crop": np.zeros((4, 4, 3))}]

    pipeline = LiveVideoPipeline(
        factory, config=fast_config(), on_metadata=received.append, processor=leaking
    )
    pipeline.add_source(src("cam-a"))
    pipeline.start()
    try:
        assert wait_until(lambda: pipeline.stream_health("cam-a").callback_errors >= 2)
        assert received == []
    finally:
        pipeline.stop()


def test_no_agent_message_contains_frames_or_encoded_buffers(factory):
    """Todo lo que el pipeline expone hacia el bus es metadata serializable."""
    metas: list[FrameMetadata] = []
    statuses: list[StreamHealth] = []
    pipeline = LiveVideoPipeline(
        factory,
        config=fast_config(),
        on_metadata=metas.append,
        on_status=statuses.append,
    )
    pipeline.add_source(src("cam-a"))
    pipeline.start()
    try:
        assert wait_until(lambda: len(metas) >= 3 and len(statuses) >= 1)
    finally:
        pipeline.stop()

    envelopes = [
        MetadataEnvelope(
            source="pipeline", stream_id=m.stream_id, payload=m.to_payload()
        )
        for m in metas
    ] + [
        MetadataEnvelope(
            source="pipeline", stream_id=s.stream_id, payload=s.to_payload()
        )
        for s in statuses
    ]
    envelopes.append(MetadataEnvelope(source="pipeline", payload=pipeline.health()))
    for envelope in envelopes:
        ensure_metadata_only(envelope.payload)
        assert not any(
            isinstance(v, (bytes, bytearray, memoryview, np.ndarray))
            for v in envelope.payload.values()
        )


def test_status_callback_reports_state_transitions(factory):
    statuses: list[StreamHealth] = []
    pipeline = LiveVideoPipeline(
        factory, config=fast_config(), on_status=statuses.append
    )
    pipeline.add_source(src("cam-a"))
    pipeline.start()
    assert wait_until(lambda: state_of(pipeline, "cam-a") is StreamState.CONNECTED)
    pipeline.stop()
    states = [s.state for s in statuses]
    assert StreamState.CONNECTING in states
    assert StreamState.CONNECTED in states
    assert states[-1] is StreamState.STOPPED
