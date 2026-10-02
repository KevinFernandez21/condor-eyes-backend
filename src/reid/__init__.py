"""Re-identificación de personas entre cámaras para tracking continuo (issue #11)."""

from __future__ import annotations

from .associate import (
    AssociationResult,
    AssociatorConfig,
    CrossCameraAssociator,
    LinkStatus,
    TrackDescriptor,
    Transition,
)

__all__ = [
    "AssociationResult",
    "AssociatorConfig",
    "CrossCameraAssociator",
    "LinkStatus",
    "TrackDescriptor",
    "Transition",
]
