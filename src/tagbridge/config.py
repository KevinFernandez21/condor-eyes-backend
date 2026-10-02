"""Site map: zonas, receptor de cada zona y cámara/región que la ve (`configs/site_map.toml`)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SiteMapError(ValueError):
    """El site map es inválido."""


@dataclass(frozen=True)
class Camera:
    id: str
    source: int | str


@dataclass(frozen=True)
class ReceiverCfg:
    id: str
    kind: str
    zone: str


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    receiver: str
    camera: str
    region: tuple[float, float, float, float]  # x0, y0, x1, y1 normalizados [0, 1]
    enter_dbm: float
    hysteresis_db: float
    calibrated: bool


@dataclass(frozen=True)
class SiteMap:
    cameras: dict[str, Camera]
    receivers: dict[str, ReceiverCfg]
    zones: dict[str, Zone]

    def zone(self, zone_id: str) -> Zone:
        try:
            return self.zones[zone_id]
        except KeyError:
            raise SiteMapError(f"zona desconocida: {zone_id!r}") from None

    def receiver(self, receiver_id: str) -> ReceiverCfg:
        try:
            return self.receivers[receiver_id]
        except KeyError:
            raise SiteMapError(f"receptor desconocido: {receiver_id!r}") from None

    def camera(self, camera_id: str) -> Camera:
        try:
            return self.cameras[camera_id]
        except KeyError:
            raise SiteMapError(f"cámara desconocida: {camera_id!r}") from None


def _unique(items: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if "id" not in item:
            raise SiteMapError(f"{kind} sin 'id'")
        if item["id"] in out:
            raise SiteMapError(f"{kind} duplicado/a: {item['id']!r}")
        out[item["id"]] = item
    return out


def _region(value: Any, zone_id: str) -> tuple[float, float, float, float]:
    ok = (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    )
    if ok:
        x0, y0, x1, y1 = (float(v) for v in value)
        ok = 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    if not ok:
        raise SiteMapError(
            f"región inválida en la zona {zone_id!r}: se espera [x0, y0, x1, y1] "
            "normalizado con x0 < x1 e y0 < y1"
        )
    return (x0, y0, x1, y1)


def load_site_map(path: str | Path) -> SiteMap:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    cameras = {
        k: Camera(id=k, source=v.get("source", 0))
        for k, v in _unique(raw.get("cameras", []), "cámara").items()
    }
    receivers = {
        k: ReceiverCfg(id=k, kind=v.get("kind", "pc-ble"), zone=v.get("zone", ""))
        for k, v in _unique(raw.get("receivers", []), "receptor").items()
    }
    zones: dict[str, Zone] = {}
    for k, v in _unique(raw.get("zones", []), "zona").items():
        if v.get("receiver") not in receivers:
            raise SiteMapError(
                f"zona {k!r}: receptor desconocido {v.get('receiver')!r}"
            )
        if v.get("camera") not in cameras:
            raise SiteMapError(f"zona {k!r}: cámara desconocida {v.get('camera')!r}")
        zones[k] = Zone(
            id=k,
            name=v.get("name", k),
            receiver=v["receiver"],
            camera=v["camera"],
            region=_region(v.get("region"), k),
            enter_dbm=float(v["enter_dbm"]),
            hysteresis_db=float(v.get("hysteresis_db", 6.0)),
            calibrated=bool(v.get("calibrated", False)),
        )
    for r in receivers.values():
        if r.zone not in zones:
            raise SiteMapError(f"receptor {r.id!r}: zona desconocida {r.zone!r}")
    return SiteMap(cameras=cameras, receivers=receivers, zones=zones)
