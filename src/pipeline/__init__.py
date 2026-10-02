"""Contratos del plano de video compartido."""

from .buffer import DropOldestQueue
from .health import BackoffPolicy, StreamHealth, StreamState
from .metadata import FrameMetadata, ensure_metadata_only
from .shared_pipeline import SharedVideoPipeline, SourceKind, StreamSource
from .trt_engine import TensorRTEngineSpec

__all__ = [
    "BackoffPolicy",
    "DropOldestQueue",
    "FrameMetadata",
    "SharedVideoPipeline",
    "SourceKind",
    "StreamHealth",
    "StreamSource",
    "StreamState",
    "TensorRTEngineSpec",
    "ensure_metadata_only",
]
