"""Credenciales RTSP: nunca deben aparecer en errores, logs ni salud."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from pipeline import (
    BackoffPolicy,
    LiveVideoPipeline,
    OpenCVFrameSource,
    PipelineConfig,
    SourceError,
    StreamSource,
    redact_credentials,
    sanitize_uri,
)

SECRET_URI = "rtsp://admin:s3cr3t@10.0.0.5:554/stream1?ch=1"


def test_sanitize_uri_strips_userinfo_but_keeps_the_rest():
    assert sanitize_uri(SECRET_URI) == "rtsp://10.0.0.5:554/stream1?ch=1"
    assert sanitize_uri("rtsp://user@host/x") == "rtsp://host/x"
    assert sanitize_uri("rtsps://cam.local/live") == "rtsps://cam.local/live"
    assert sanitize_uri("/dev/video0") == "/dev/video0"
    assert sanitize_uri("usb:1") == "usb:1"


def test_redact_credentials_in_free_text():
    text = f"fallo abriendo {SECRET_URI} y también rtsp://a:b@h2/x"
    cleaned = redact_credentials(text)
    assert "s3cr3t" not in cleaned and "admin" not in cleaned and ":b@" not in cleaned
    assert "10.0.0.5:554/stream1" in cleaned


def test_stream_source_repr_and_safe_uri_hide_credentials():
    source = StreamSource("cam", SECRET_URI)
    assert "s3cr3t" not in repr(source)
    assert "s3cr3t" not in source.safe_uri
    assert source.uri == SECRET_URI  # el backend sí necesita la URI real


class FailingCapture:
    def __init__(self, target) -> None:
        pass

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        pass


def test_opencv_errors_do_not_leak_credentials(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "cv2", types.SimpleNamespace(VideoCapture=FailingCapture)
    )
    with pytest.raises(SourceError) as info:
        OpenCVFrameSource(StreamSource("cam", SECRET_URI)).open()
    assert "s3cr3t" not in str(info.value)
    assert "10.0.0.5" in str(info.value)


def test_pipeline_health_and_logs_never_contain_credentials(caplog):
    class LeakyFactory:
        def __call__(self, source):
            raise RuntimeError(f"timeout conectando a {source.uri}")

    pipeline = LiveVideoPipeline(
        LeakyFactory(),
        config=PipelineConfig(
            backoff=BackoffPolicy(initial=0.01, maximum=0.02), max_retries=2
        ),
    )
    pipeline.add_source(StreamSource("cam", SECRET_URI))
    with caplog.at_level(logging.DEBUG):
        pipeline.start()
        import time

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if pipeline.stream_health("cam").last_error:
                break
            time.sleep(0.005)
        pipeline.stop()

    health = pipeline.stream_health("cam")
    assert health.last_error and "s3cr3t" not in health.last_error
    assert "s3cr3t" not in str(health.to_payload())
    assert "s3cr3t" not in str(pipeline.health())
    assert "s3cr3t" not in caplog.text
