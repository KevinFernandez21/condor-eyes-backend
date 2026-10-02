"""Servicio de localización: ingesta, validación, estimación y autorización.

Orquesta `payload` -> repositorio (tag enrolado) -> `ObservationValidator` ->
`ZoneEstimator`. Todo reloj es inyectable para pruebas deterministas. Los logs
solo contienen seudónimos; ver `privacy.py`.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from .config import LocationConfig
from .estimator import ZoneEstimator
from .models import EstimateStatus, RejectReason, TagObservation, ZoneEstimate
from .payload import PayloadError, parse_binary, parse_json
from .privacy import Pseudonymizer
from .repository import READ_SCOPE, PersonnelRepository, Principal
from .validation import ObservationValidator

log = logging.getLogger("condor.location")

SERVICE_PRINCIPAL = Principal("location-service", frozenset({READ_SCOPE}))


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Resultado de procesar un paquete."""

    accepted: bool
    reason: RejectReason | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class LocationReport:
    """Estimación enriquecida con la autorización de la persona (sin datos en claro)."""

    estimate: ZoneEstimate
    person_ref: str | None
    authorized: bool | None


class LocationService:
    """Punto de entrada del componente de localización."""

    def __init__(
        self,
        config: LocationConfig,
        repository: PersonnelRepository,
        pseudonymizer: Pseudonymizer,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        principal: Principal = SERVICE_PRINCIPAL,
    ) -> None:
        self._cfg = config
        self._repo = repository
        self._pseudo = pseudonymizer
        self._clock = clock
        self._principal = principal
        self._validator = ObservationValidator(config)
        self._estimator = ZoneEstimator(config, pseudonymizer)
        self._stats: Counter[RejectReason] = Counter()
        self._accepted = 0
        self._last_prune: datetime | None = None

    @property
    def stats(self) -> Counter[RejectReason]:
        """Contadores de paquetes rechazados por motivo."""
        return self._stats

    @property
    def accepted_count(self) -> int:
        return self._accepted

    # --- ingesta ---------------------------------------------------------

    def ingest_json(
        self, raw: str | bytes, received_at: datetime | None = None
    ) -> IngestResult:
        try:
            obs = parse_json(raw)
        except PayloadError as exc:
            return self._malformed(exc)
        return self.ingest(obs, received_at)

    def ingest_binary(
        self, raw: bytes, received_at: datetime | None = None
    ) -> IngestResult:
        try:
            obs = parse_binary(raw)
        except PayloadError as exc:
            return self._malformed(exc)
        return self.ingest(obs, received_at)

    def ingest(
        self, obs: TagObservation, received_at: datetime | None = None
    ) -> IngestResult:
        """Valida y registra una observación. Nunca lanza por datos inválidos."""
        now = received_at or self._clock()
        tag_ref = self._pseudo.ref("tag", obs.tag_id)

        if self._repo.lookup(obs.tag_id, requester=self._principal) is None:
            return self._reject(RejectReason.UNENROLLED_TAG, tag_ref, obs)

        reason = self._validator.validate(obs, now)
        if reason is not None:
            # Un nodo que habla sigue vivo aunque su dato se descarte (salvo nodo desconocido).
            if reason is not RejectReason.UNKNOWN_NODE:
                self._estimator.touch_node(obs.node_id, now)
            return self._reject(reason, tag_ref, obs)

        self._estimator.add(obs, now)
        self._accepted += 1
        self._maybe_prune(now)
        return IngestResult(True)

    def heartbeat(self, node_id: str, at: datetime | None = None) -> None:
        """Registra que un nodo está vivo (mensaje de latido sin tags)."""
        if self._cfg.zone_of_node(node_id) is None:
            log.debug("latido de nodo desconocido ignorado: nodo=%s", node_id)
            return
        self._estimator.touch_node(node_id, at or self._clock())

    # --- consulta --------------------------------------------------------

    def estimate(self, tag_id: str, now: datetime | None = None) -> ZoneEstimate:
        return self._estimator.estimate(tag_id, now or self._clock())

    def reports(self, now: datetime | None = None) -> list[LocationReport]:
        """Una estimación por cada tag enrolado (los ausentes salen como UNKNOWN)."""
        instant = now or self._clock()
        self._estimator.prune(instant)
        self._validator.prune(instant)
        reports: list[LocationReport] = []
        for tag_id in sorted(self._repo.enrolled_tags(requester=self._principal)):
            estimate = self._estimator.estimate(tag_id, instant)
            record = self._repo.lookup(tag_id, requester=self._principal)
            person_ref = (
                self._pseudo.ref("person", record.person_id) if record else None
            )
            authorized: bool | None = None
            if record is not None and estimate.status is EstimateStatus.LOCATED:
                authorized = estimate.zone_id in record.allowed_zones
            reports.append(LocationReport(estimate, person_ref, authorized))
        log.debug("estimaciones calculadas: total=%d", len(reports))
        return reports

    # --- internos --------------------------------------------------------

    def _malformed(self, exc: PayloadError) -> IngestResult:
        self._stats[RejectReason.MALFORMED] += 1
        log.warning("paquete malformado descartado: %s", exc)
        return IngestResult(False, RejectReason.MALFORMED, str(exc))

    def _reject(
        self, reason: RejectReason, tag_ref: str, obs: TagObservation
    ) -> IngestResult:
        self._stats[reason] += 1
        log.info(
            "observación rechazada: motivo=%s tag=%s nodo=%s seq=%d",
            reason.value,
            tag_ref,
            obs.node_id,
            obs.sequence,
        )
        return IngestResult(False, reason)

    def _maybe_prune(self, now: datetime) -> None:
        interval = self._cfg.retention_s / 10
        if (
            self._last_prune is None
            or (now - self._last_prune).total_seconds() > interval
        ):
            self._last_prune = now
            self._estimator.prune(now)
            self._validator.prune(now)
