"""Fusión de evidencia: visión + identidad + ubicación + permisos (issue #15)."""

from .adapters import identity_from_envelope, reid_from_envelope
from .config import FusionConfig, load_fusion_config
from .decision import (
    DecisionOutcome,
    DecisionRecord,
    EvidenceKind,
    EvidenceRef,
    EvidenceRole,
    ReasonCode,
)
from .engine import FusionEngine
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
    "DecisionOutcome",
    "DecisionRecord",
    "EvidenceKind",
    "EvidenceRef",
    "EvidenceRole",
    "FusionConfig",
    "FusionEngine",
    "FusionInput",
    "IdentityEvidence",
    "IdentityStatus",
    "InMemoryPermissions",
    "LocationEvidence",
    "PermissionRepository",
    "ReasonCode",
    "ReidLink",
    "ReidStatus",
    "TrackObservation",
    "ZoneRule",
    "identity_from_envelope",
    "load_fusion_config",
    "reid_from_envelope",
]
