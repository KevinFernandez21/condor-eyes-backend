"""Contratos del plano de video: fuentes, salud, backoff, buffer y metadata."""

from __future__ import annotations

import random
from datetime import UTC, datetime

import numpy as np
import pytest

from bus import MetadataEnvelope
from pipeline import (
    BackoffPolicy,
    DropOldestQueue,
    FrameMetadata,
    SourceKind,
    StreamHealth,
    StreamSource,
    StreamState,
    ensure_metadata_only,
)


@pytest.mark.parametrize(
    "uri",
    ["rtsp://10.0.0.5:554/stream1", "rtsps://cam.local/live"],
)
def test_rtsp_uris_are_classified(uri):
    assert StreamSource("cam-01", uri).kind is SourceKind.RTSP


@pytest.mark.parametrize(
    "uri",
    ["/dev/video0", "v4l2:///dev/video2", "usb:0", "usb://1"],
)
def test_usb_uris_are_classified(uri):
    assert StreamSource("cam-usb", uri).kind is SourceKind.USB


@pytest.mark.parametrize(
    ("stream_id", "uri"),
    [
        ("", "rtsp://host/x"),
        ("   ", "rtsp://host/x"),
        ("cam-01", "http://host/x"),
        ("cam-01", "rtsp://"),
        ("cam-01", "usb:abc"),
        ("cam-01", ""),
    ],
)
def test_invalid_sources_are_rejected_in_spanish(stream_id, uri):
    with pytest.raises(ValueError, match="stream_id|URI"):
        StreamSource(stream_id, uri)


def test_backoff_is_exponential_and_capped():
    policy = BackoffPolicy(initial=1.0, factor=2.0, maximum=8.0)
    assert [policy.delay(n) for n in range(6)] == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_backoff_jitter_stays_within_bounds():
    policy = BackoffPolicy(initial=2.0, factor=2.0, maximum=30.0, jitter=0.25)
    rng = random.Random(7)
    for attempt in range(8):
        base = min(2.0 * 2.0**attempt, 30.0)
        delay = policy.delay(attempt, rng=rng)
        assert base * 0.75 <= delay <= base * 1.25


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial": 0},
        {"factor": 0.5},
        {"maximum": 0.1, "initial": 1.0},
        {"jitter": 1.5},
    ],
)
def test_backoff_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        BackoffPolicy(**kwargs)


def test_drop_oldest_queue_is_bounded_and_counts_drops():
    queue: DropOldestQueue[int] = DropOldestQueue(maxsize=3)
    dropped = [queue.put(n) for n in range(6)]
    assert dropped == [0, 0, 0, 1, 1, 1]
    assert len(queue) == 3
    assert [queue.get_nowait() for _ in range(3)] == [3, 4, 5]
    assert queue.get_nowait() is None


def test_drop_oldest_queue_clear_and_invalid_size():
    queue: DropOldestQueue[int] = DropOldestQueue(maxsize=2)
    queue.put(1)
    queue.clear()
    assert len(queue) == 0
    with pytest.raises(ValueError, match="maxsize"):
        DropOldestQueue(maxsize=0)


def test_stream_health_payload_has_required_fields_and_is_serializable():
    health = StreamHealth(
        stream_id="cam-01",
        state=StreamState.RECONNECTING,
        last_frame_at=datetime(2026, 1, 1, tzinfo=UTC),
        retry_count=3,
        last_error="ConnectionError: timeout",
    )
    payload = health.to_payload()
    assert payload["state"] == "reconnecting"
    assert payload["retry_count"] == 3
    assert payload["last_error"] == "ConnectionError: timeout"
    assert payload["last_frame_at"] == "2026-01-01T00:00:00+00:00"
    ensure_metadata_only(payload)


@pytest.mark.parametrize(
    "bad",
    [
        b"\xff\xd8jpeg",
        bytearray(b"abc"),
        memoryview(b"abc"),
        np.zeros((2, 2, 3), dtype=np.uint8),
        {"nested": [{"frame": np.zeros(4)}]},
        {"crop": b"raw"},
    ],
)
def test_ensure_metadata_only_rejects_frames_and_encoded_buffers(bad):
    with pytest.raises(TypeError, match="frame|buffer|bytes"):
        ensure_metadata_only(bad)


def test_frame_metadata_travels_in_a_metadata_envelope_without_pixels():
    meta = FrameMetadata(
        stream_id="cam-01",
        frame_index=42,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        width=1920,
        height=1080,
        detections=({"label": "person", "conf": 0.9, "bbox": [1, 2, 3, 4]},),
    )
    payload = meta.to_payload()
    ensure_metadata_only(payload)
    envelope = MetadataEnvelope(source="pipeline", stream_id="cam-01", payload=payload)
    assert envelope.payload["frame_index"] == 42
    assert envelope.payload["width"] == 1920
    assert "frame" not in envelope.payload and "data" not in envelope.payload


def test_frame_metadata_rejects_pixel_data_in_detections():
    with pytest.raises(TypeError):
        FrameMetadata(
            stream_id="cam-01",
            frame_index=1,
            timestamp=datetime.now(UTC),
            width=2,
            height=2,
            detections=({"crop": np.zeros((2, 2, 3))},),
        )
