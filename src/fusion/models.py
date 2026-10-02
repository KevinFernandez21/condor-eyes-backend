"""Entradas tipadas de la fusión: solo metadata, nunca biometría ni imágenes.

Cada evidencia lleva un `evidence_id` que identifica el mensaje de origen (el que
el productor publicó por el bus). Es la clave para trazar una decisión hasta su fuente.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol


def _check_aware(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} debe incluir zona horaria")


def _check_unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} debe estar en [0, 1]")


class IdentityStatus(StrEnum):
    """Estados de la verificación facial (issue #10)."""

    MATCH = "match"
    UNKNOWN = "unknown"
    INCONCLUSIVE = "inconclusive"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    INVALID_INPUT = "invalid_input"


class ReidStatus(StrEnum):
    """Estados de la re-identificación entre cámaras (issue #11)."""

    LINKED = "linked"
    INCONCLUSIVE = "inconclusive"
    NO_CANDIDATE = "no_candidate"


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """Persona rastreada en una cámara y la zona que cubre."""

    evidence_id: str
    track_ref: str  # único entre cámaras, p. ej. "cam1/7"
    stream_id: str
    zone_id: str
    observed_at: datetime
    confidence: float

    def __post_init__(self) -> None:
        _check_aware("observed_at", self.observed_at)
        _check_unit("confidence", self.confidence)


@dataclass(frozen=True, slots=True)
class IdentityEvidence:
    """Resultado de verificación facial para una pista. No contiene embeddings."""

    evidence_id: str
    track_ref: str
    status: IdentityStatus
    observed_at: datetime
    person_id: str | None = None
    score: float = 0.0

    def __post_init__(self) -> None:
        _check_aware("observed_at", self.observed_at)
        _check_unit("score", self.score)


@dataclass(frozen=True, slots=True)
class ReidLink:
    """Asociación entre una pista y otra de otra cámara."""

    evidence_id: str
    source_track_ref: str
    candidate_track_ref: str | None
    status: ReidStatus
    confidence: float
    observed_at: datetime

    def __post_init__(self) -> None:
        _check_aware("observed_at", self.observed_at)
        _check_unit("confidence", self.confidence)


@dataclass(frozen=True, slots=True)
class LocationEvidence:
    """Zona estimada de un tag de personal. `person_id` ya viene resuelto (sin ID de tag).

    `valid=False` marca paquetes duplicados, repetidos o imposibles detectados aguas arriba.
    """

    evidence_id: str
    person_id: str
    zone_id: str
    observed_at: datetime
    confidence: float
    valid: bool = True

    def __post_init__(self) -> None:
        _check_aware("observed_at", self.observed_at)
        _check_unit("confidence", self.confidence)


@dataclass(frozen=True, slots=True)
class ZoneRule:
    """Regla de zona. En una zona no restringida no se exige corroboración."""

    zone_id: str
    restricted: bool = True


class PermissionRepository(Protocol):
    """Repositorio intercambiable de permisos por persona."""

    def allowed_zones(self, person_id: str) -> frozenset[str] | None:
        """Zonas permitidas, o None si no existe registro de la persona."""


class InMemoryPermissions:
    """Implementación en memoria para pruebas y prototipos."""

    def __init__(self, zones_by_person: Mapping[str, frozenset[str]]) -> None:
        self._zones = dict(zones_by_person)

    def allowed_zones(self, person_id: str) -> frozenset[str] | None:
        return self._zones.get(person_id)


@dataclass(frozen=True, slots=True)
class FusionInput:
    """Evidencia disponible en un instante de evaluación."""

    tracks: tuple[TrackObservation, ...] = ()
    identities: tuple[IdentityEvidence, ...] = ()
    reids: tuple[ReidLink, ...] = ()
    locations: tuple[LocationEvidence, ...] = ()
