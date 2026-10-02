"""Fusión de evidencia: visión + identidad + ubicación + permisos (issue #15)."""
from .config import FusionConfig, load_fusion_config
from .models import (
    FusionInput,
    IdentityEvidence,
    IdentityStatus,
    InMemoryPermissions,
    LocationEvidence,
    PermissionRepository,
    ReidLink,
    ReidStatus,
    TrackObservation,
    ZoneRule,
)

__all__ = [
    "FusionConfig",
    "FusionInput",
    "IdentityEvidence",
    "IdentityStatus",
    "InMemoryPermissions",
    "LocationEvidence",
    "PermissionRepository",
    "ReidLink",
    "ReidStatus",
    "TrackObservation",
    "ZoneRule",
    "load_fusion_config",
]
