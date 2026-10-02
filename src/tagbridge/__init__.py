"""Puente del tag BLE (XIAO ESP32-C6) hacia observaciones payload v1 (issue #26)."""

from .config import SiteMap, SiteMapError, Zone, load_site_map
from .presence import PresenceState, PresenceView
from .protocol import TagPayload, decode_tag_payload, encode_tag_payload, normalize_tag
from .receiver import (
    Advertisement,
    ReceiverEvent,
    SourceClosed,
    TagReceiver,
    run_receiver,
)
from .scanner import BleakAdvertisementSource, FakeAdvertisementSource

__all__ = [
    "Advertisement",
    "BleakAdvertisementSource",
    "FakeAdvertisementSource",
    "PresenceState",
    "PresenceView",
    "ReceiverEvent",
    "SiteMap",
    "SiteMapError",
    "SourceClosed",
    "TagPayload",
    "TagReceiver",
    "Zone",
    "decode_tag_payload",
    "encode_tag_payload",
    "load_site_map",
    "normalize_tag",
    "run_receiver",
]
