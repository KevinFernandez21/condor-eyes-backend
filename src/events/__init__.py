"""Detección de eventos (merodeo, intrusión en zona, aglomeración) sobre tracks."""

from __future__ import annotations

from .config import load_events_config
from .engine import EventEngine, LoiterFeatures, point_in_polygon, rule_loiter
from .model import (
    EVENT_TOPIC,
    AuthorizedPolicy,
    Event,
    EventPolicy,
    EventType,
    GroundTruthEvent,
    Severity,
    TrackObservation,
    Zone,
)

__all__ = [
    "EVENT_TOPIC",
    "AuthorizedPolicy",
    "Event",
    "EventEngine",
    "EventPolicy",
    "EventType",
    "GroundTruthEvent",
    "LoiterFeatures",
    "Severity",
    "TrackObservation",
    "Zone",
    "load_events_config",
    "point_in_polygon",
    "rule_loiter",
]
