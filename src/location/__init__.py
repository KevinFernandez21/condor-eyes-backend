"""Localización de personal por zonas a partir de tags BLE/LoRa.

Ver `docs/location.md` para el contrato de payload y el algoritmo.
"""

from .config import ConfigError, LocationConfig, load_config, parse_config
from .models import (
    Channel,
    EstimateStatus,
    Evidence,
    RejectReason,
    TagObservation,
    UnknownReason,
    ZoneEstimate,
)
from .payload import PayloadError, parse_binary, parse_json
from .privacy import Pseudonymizer
from .repository import (
    AccessDenied,
    InMemoryPersonnelRepository,
    PersonnelRecord,
    PersonnelRepository,
    Principal,
)
from .service import IngestResult, LocationReport, LocationService

__all__ = [
    "AccessDenied",
    "Channel",
    "ConfigError",
    "EstimateStatus",
    "Evidence",
    "InMemoryPersonnelRepository",
    "IngestResult",
    "LocationConfig",
    "LocationReport",
    "LocationService",
    "PayloadError",
    "PersonnelRecord",
    "PersonnelRepository",
    "Principal",
    "Pseudonymizer",
    "RejectReason",
    "TagObservation",
    "UnknownReason",
    "ZoneEstimate",
    "load_config",
    "parse_binary",
    "parse_config",
    "parse_json",
]
