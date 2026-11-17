"""Utilidades compartidas por las pruebas de localización."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from location.config import LocationConfig, parse_config
from location.models import Channel, TagObservation

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
TAG = "A1B2C3D4E5F6"
TAG2 = "0A0B0C0D0E0F"

ZONES = {
    "lobby": {"nodes": ["N0001"], "neighbors": ["hall"]},
    "hall": {"nodes": ["N0002", "N0003"], "neighbors": ["lobby", "lab"]},
    "lab": {"nodes": ["N0004"], "neighbors": ["hall"]},
}

# Posiciones (metros) de los nodos en una línea: lobby - hall - lab.
NODE_POSITIONS = {
    "N0001": (0.0, 0.0),
    "N0002": (10.0, 0.0),
    "N0003": (14.0, 0.0),
    "N0004": (24.0, 0.0),
}


def make_config(**sections) -> LocationConfig:
    return parse_config({"zones": ZONES, **sections})


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def obs(
    *,
    node: str = "N0001",
    rssi: int = -60,
    t: float = 0.0,
    seq: int = 1,
    tag: str = TAG,
    battery: int | None = 90,
    channel: Channel = Channel.BLE,
) -> TagObservation:
    return TagObservation(
        tag_id=tag,
        node_id=node,
        rssi_dbm=rssi,
        timestamp=at(t),
        sequence=seq,
        battery_pct=battery,
        channel=channel,
    )
