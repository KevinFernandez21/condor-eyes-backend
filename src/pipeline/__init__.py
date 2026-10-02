"""Contratos del plano de video compartido."""

from .buffer import DropOldestQueue
from .fake import FakeCamera, FakeFrameSource, FakeSourceFactory
from .health import BackoffPolicy, StreamHealth, StreamState
from .live_pipeline import LiveVideoPipeline, PipelineConfig
from .metadata import FrameMetadata, ensure_metadata_only
from .shared_pipeline import SharedVideoPipeline, SourceKind, StreamSource
from .sources import (
    Frame,
    FrameSource,
    OpenCVFrameSource,
    SourceError,
    SourceFactory,
    opencv_source_factory,
)
from .trt_engine import TensorRTEngineSpec

__all__ = [
    "BackoffPolicy",
    "DropOldestQueue",
    "FakeCamera",
    "FakeFrameSource",
    "FakeSourceFactory",
    "Frame",
    "FrameMetadata",
    "FrameSource",
    "LiveVideoPipeline",
    "OpenCVFrameSource",
    "PipelineConfig",
    "SharedVideoPipeline",
    "SourceError",
    "SourceFactory",
    "SourceKind",
    "StreamHealth",
    "StreamSource",
    "StreamState",
    "TensorRTEngineSpec",
    "ensure_metadata_only",
    "opencv_source_factory",
]
