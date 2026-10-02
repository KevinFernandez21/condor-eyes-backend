"""Metadata por frame y guardia que impide que pixeles lleguen al bus."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

_BUFFER_TYPES = (bytes, bytearray, memoryview)


def ensure_metadata_only(value: object, _path: str = "payload") -> None:
    """Lanza TypeError si `value` contiene frames, arrays o buffers binarios."""
    if isinstance(value, _BUFFER_TYPES):
        raise TypeError(
            f"{_path}: los buffers binarios (bytes) no pueden viajar por el bus"
        )
    if hasattr(value, "__array__") or hasattr(value, "shape"):
        raise TypeError(f"{_path}: un frame o array no puede viajar por el bus")
    if isinstance(value, Mapping):
        for key, item in value.items():
            ensure_metadata_only(item, f"{_path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for index, item in enumerate(value):
            ensure_metadata_only(item, f"{_path}[{index}]")


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Resultado por frame que el pipeline entrega a los agentes."""

    stream_id: str
    frame_index: int
    timestamp: datetime
    width: int
    height: int
    detections: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        ensure_metadata_only(self.detections, "detections")

    def to_payload(self) -> dict[str, Any]:
        """Payload serializable para un MetadataEnvelope (sin pixeles)."""
        return {
            "stream_id": self.stream_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.isoformat(),
            "width": self.width,
            "height": self.height,
            "detections": [dict(d) for d in self.detections],
        }
