"""Frontera tipada del bus de metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .hub import (
    DEFAULT_PAYLOAD_VERSION,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    HubError,
    InvalidEnvelopeError,
    InvalidTopicError,
    MetadataEnvelope,
    MetadataHub,
    MetadataSubscription,
    Topic,
    UnsupportedVersionError,
    build_error_envelope,
    envelope_from_dict,
    envelope_to_dict,
    parse_topic,
    register_payload_versions,
    supported_payload_versions,
    validate_envelope,
)
from .memory import HubClosedError, InMemoryHub

if TYPE_CHECKING:
    from .agentscope_hub import AgentScopeHub


def __getattr__(name: str) -> Any:
    """Exporta ``AgentScopeHub`` de forma diferida (PEP 562)."""
    if name == "AgentScopeHub":
        from .agentscope_hub import AgentScopeHub

        return AgentScopeHub
    raise AttributeError(f"module 'bus' has no attribute {name!r}")


__all__ = [
    "DEFAULT_PAYLOAD_VERSION",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "AgentScopeHub",
    "HubClosedError",
    "HubError",
    "InMemoryHub",
    "InvalidEnvelopeError",
    "InvalidTopicError",
    "MetadataEnvelope",
    "MetadataHub",
    "MetadataSubscription",
    "Topic",
    "UnsupportedVersionError",
    "build_error_envelope",
    "envelope_from_dict",
    "envelope_to_dict",
    "parse_topic",
    "register_payload_versions",
    "supported_payload_versions",
    "validate_envelope",
]
