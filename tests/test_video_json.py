"""Los payloads del publisher deben ser JSON estricto (contrato de MetadataEnvelope)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from test_video_publisher import RecordingHub, loop, wait_until  # noqa: F401

from bus import MetadataEnvelope, Topic
from pipeline import (
    BackoffPolicy,
    FakeSourceFactory,
    LiveVideoPipeline,
    MetadataPublisher,
    PipelineConfig,
    StreamSource,
)


@pytest.mark.parametrize(
    "bad",
    [
        {"when": datetime(2026, 1, 1, tzinfo=UTC)},
        {"tags": {"a", "b"}},
        {"obj": object()},
        {"n": np.float32(0.5)},
        {1: "clave no str"},
        {"nested": [{"x": {"y"}}]},
    ],
)
def test_publisher_rejects_non_json_payloads(loop, bad):  # noqa: F811
    hub = RecordingHub()
    publisher = MetadataPublisher(hub, loop)
    with pytest.raises(TypeError):
        publisher.publish_payload(Topic.EVENTS, "cam-01", bad)
    assert hub.published == []


def test_every_published_envelope_is_json_serializable(loop):  # noqa: F811
    hub = RecordingHub()
    publisher = MetadataPublisher(hub, loop)
    factory = FakeSourceFactory(default_interval=0.002)
    factory.camera("cam-b").disconnect()
    pipeline = LiveVideoPipeline(
        factory,
        config=PipelineConfig(
            backoff=BackoffPolicy(initial=0.01, maximum=0.05),
            watchdog_timeout=1.0,
            watchdog_interval=0.05,
        ),
        publisher=publisher,
        processor=lambda stream_id, frame: [
            {"label": "person", "conf": 0.9, "bbox": (1, 2, 3, 4)}
        ],
    )
    pipeline.add_source(StreamSource("cam-a", "rtsp://fake/a"))
    pipeline.add_source(StreamSource("cam-b", "rtsp://fake/b"))
    pipeline.start()
    try:
        assert wait_until(
            lambda: (
                sum(t is Topic.DETECTIONS for t, _ in hub.published) >= 3
                and any(
                    e.payload.get("last_error")
                    for t, e in hub.published
                    if t is Topic.STREAM_STATUS
                )
            )
        )
    finally:
        pipeline.stop()
    envelopes = [envelope for _, envelope in hub.published]
    envelopes.append(MetadataEnvelope("pipeline", pipeline.health()))
    assert len(envelopes) >= 4
    for envelope in envelopes:
        json.dumps(envelope.payload)  # no debe lanzar
        assert isinstance(envelope.payload.get("timestamp", ""), str)
