"""Pruebas de los límites arquitectónicos del scaffold multiagente."""

from pathlib import Path

import pytest

from agents import MULTIAGENT_ROUTE, get_route, validate_route
from bus import MetadataEnvelope, Topic
from pipeline import TensorRTEngineSpec


def test_route_contains_the_seven_architecture_roles():
    assert {route.name for route in MULTIAGENT_ROUTE} == {
        "ingest",
        "inference",
        "tracker",
        "event",
        "storage",
        "supervisor",
        "comms",
    }
    validate_route()


def test_inference_is_singleton_and_ingest_scales_per_camera():
    assert get_route("inference").singleton
    assert not get_route("ingest").singleton


def test_bus_exposes_only_metadata_topics():
    assert all(
        "frame" not in topic.value and "video" not in topic.value for topic in Topic
    )
    envelope = MetadataEnvelope(
        source="tracker", stream_id="cam-01", payload={"track_id": 7}
    )
    assert envelope.payload == {"track_id": 7}


def test_tensorrt_engine_requires_fp16_engine_artifact():
    spec = TensorRTEngineSpec(Path("models/yolov8n-v1.engine"), "v1", batch_size=4)
    assert spec.precision == "fp16"

    with pytest.raises(ValueError, match="FP16"):
        TensorRTEngineSpec(
            Path("models/yolov8n-v1.engine"), "v1", batch_size=4, precision="fp32"
        )
