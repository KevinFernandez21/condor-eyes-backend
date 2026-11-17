"""Repositorio reemplazable que mapea tags a personal y permisos.

La implementación real (SQLite, LDAP, API de RR. HH.) debe respetar el mismo
`Protocol`: todo acceso exige un `Principal` con el alcance `personnel:read`,
y los accesos se auditan solo con seudónimos, nunca con IDs en claro.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from .privacy import Pseudonymizer

READ_SCOPE = "personnel:read"

audit_log = logging.getLogger("condor.location.audit")


class AccessDenied(PermissionError):
    """El solicitante no tiene permiso para consultar el repositorio."""


@dataclass(frozen=True, slots=True)
class Principal:
    """Identidad que consulta el repositorio (servicio o usuario) y sus alcances."""

    name: str
    scopes: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PersonnelRecord:
    """Persona enrolada. Los campos personales se excluyen del `repr`."""

    person_id: str = field(repr=False)
    display_name: str = field(repr=False)
    allowed_zones: frozenset[str] = frozenset()


class PersonnelRepository(Protocol):
    """Contrato que debe cumplir cualquier almacén de personal."""

    def lookup(self, tag_id: str, *, requester: Principal) -> PersonnelRecord | None:
        """Devuelve el registro del tag o None si no está enrolado."""

    def enrolled_tags(self, *, requester: Principal) -> frozenset[str]:
        """IDs de todos los tags enrolados (para detectar tags ausentes)."""


class InMemoryPersonnelRepository:
    """Implementación en memoria para pruebas y prototipo."""

    def __init__(self, pseudonymizer: Pseudonymizer) -> None:
        self._pseudo = pseudonymizer
        self._records: dict[str, PersonnelRecord] = {}

    def enroll(self, tag_id: str, record: PersonnelRecord) -> None:
        self._records[tag_id.upper()] = record

    def _authorize(self, requester: Principal, action: str, target: str) -> None:
        if READ_SCOPE not in requester.scopes:
            audit_log.warning(
                "acceso denegado: solicitante=%s acción=%s objetivo=%s",
                requester.name,
                action,
                target,
            )
            raise AccessDenied(
                f"el solicitante '{requester.name}' no tiene el alcance '{READ_SCOPE}'"
            )

    def lookup(self, tag_id: str, *, requester: Principal) -> PersonnelRecord | None:
        key = tag_id.upper()
        tag_ref = self._pseudo.ref("tag", key)
        self._authorize(requester, "lookup", tag_ref)
        record = self._records.get(key)
        audit_log.debug(
            "consulta: solicitante=%s tag=%s encontrado=%s",
            requester.name,
            tag_ref,
            record is not None,
        )
        return record

    def enrolled_tags(self, *, requester: Principal) -> frozenset[str]:
        self._authorize(requester, "enrolled_tags", "*")
        audit_log.debug("listado de tags: solicitante=%s", requester.name)
        return frozenset(self._records)
