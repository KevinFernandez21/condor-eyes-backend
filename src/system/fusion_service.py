"""Servicio que alimenta la fusión de evidencia desde el bus y publica decisiones.

La fusión (`src/fusion`) es un motor puro; este componente la conecta al bus:
escucha tracks, ubicaciones e identidades, evalúa cada ``fusion_interval_s`` y
publica en ``Topic.EVENTS`` solo las decisiones que **cambian** (por track), de
modo que el dashboard no se inunda con repeticiones.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bus import MetadataEnvelope, MetadataHub, Topic
from fusion import (
    DecisionRecord,
    FusionConfig,
    FusionEngine,
    FusionInput,
    IdentityEvidence,
    InMemoryPermissions,
    LocationEvidence,
    TrackObservation,
    ZoneRule,
    identity_from_envelope,
    load_fusion_config,
    publish_decisions,
)

from .config import SystemConfig
from .pseudonym import Pseudonymizer
from .simulators import IDENTITY_KIND, LOCATION_KIND, LOCATION_TOPIC, PeriodicComponent

logger = logging.getLogger(__name__)

_SOURCE = "fusion"


def _unit(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if 0.0 <= number <= 1.0 else default


class FusionService(PeriodicComponent):
    """Correlaciona tracks + identidad + ubicación + permisos y emite decisiones."""

    name = "fusion"

    def __init__(
        self,
        hub: MetadataHub,
        cfg: SystemConfig,
        pseudonymizer: Pseudonymizer | None = None,
    ) -> None:
        super().__init__(hub, cfg.fusion_interval_s, status="ok")
        ps = pseudonymizer or Pseudonymizer.random()
        path = Path(cfg.fusion_config)
        if path.is_file():
            self._fusion_cfg: FusionConfig = load_fusion_config(path)
            self._detail = f"umbrales de {path.name}"
        else:
            self._fusion_cfg = FusionConfig()
            self._status = "degraded"
            self._detail = f"{path} no existe; se usan umbrales por defecto"
        self._engine = FusionEngine(
            self._fusion_cfg,
            [ZoneRule(z.zone_id, z.restricted) for z in cfg.effective_zones()],
            InMemoryPermissions(
                {ps.person(p): frozenset(z) for p, z in cfg.permissions.items()}
            ),
        )
        self._tracks: dict[str, TrackObservation] = {}
        self._identities: dict[str, IdentityEvidence] = {}
        self._locations: dict[str, LocationEvidence] = {}
        self._signatures: dict[str | None, tuple[Any, ...]] = {}
        self._subscriptions: list[Any] = []
        self.decisions = 0
        self.unknown_locations = 0

    # -- ciclo de vida --

    async def _on_start(self) -> None:
        topics = {Topic.TRACKS, Topic.HEALTH, LOCATION_TOPIC}
        for topic in sorted(topics, key=lambda t: t.value):
            subscription = self._hub.subscribe(topic)
            self._subscriptions.append(subscription)
            self._spawn(self._consume(topic, subscription))

    async def _on_stop(self) -> None:
        for subscription in self._subscriptions:
            subscription.close()
        self._subscriptions.clear()

    async def _consume(self, topic: Topic, subscription: Any) -> None:
        async for envelope in subscription:
            try:
                self._ingest(topic, envelope)
            except Exception as exc:  # noqa: BLE001 - un payload raro no tumba la fusión
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"

    # -- entrada de evidencia --

    def _ingest(self, topic: Topic, envelope: MetadataEnvelope) -> None:
        payload = envelope.payload
        if envelope.source == _SOURCE:
            return  # nuestras propias decisiones
        if topic is Topic.TRACKS:
            self._ingest_tracks(envelope)
        elif (
            payload.get("kind") == LOCATION_KIND or topic.value == "location.estimates"
        ):
            self._ingest_location(envelope)
        elif payload.get("kind") == IDENTITY_KIND:
            ref = payload.get("track_ref")
            if isinstance(ref, str):
                self._identities[ref] = identity_from_envelope(
                    envelope.event_id, ref, envelope
                )

    def _ingest_tracks(self, envelope: MetadataEnvelope) -> None:
        for track in envelope.payload.get("tracks", []):
            ref, zone = track.get("track_ref"), track.get("zone_id")
            if not isinstance(ref, str) or not isinstance(zone, str):
                continue
            if track.get("cls", 0) != 0:
                continue  # la fusión de identidad solo trata personas
            self._tracks[ref] = TrackObservation(
                evidence_id=envelope.event_id,
                track_ref=ref,
                stream_id=str(envelope.stream_id or track.get("stream_id") or "stream"),
                zone_id=zone,
                observed_at=envelope.created_at,
                confidence=_unit(track.get("conf"), 0.0),
            )

    def _ingest_location(self, envelope: MetadataEnvelope) -> None:
        payload = envelope.payload
        person, zone = payload.get("person_ref"), payload.get("zone_id")
        if payload.get("status") != "located" or not isinstance(person, str):
            self.unknown_locations += 1  # sin tag no hay evidencia: nunca se inventa
            return
        if not isinstance(zone, str):
            return
        self._locations[person] = LocationEvidence(
            evidence_id=envelope.event_id,
            person_id=person,
            zone_id=zone,
            observed_at=envelope.created_at,
            confidence=_unit(payload.get("confidence"), 0.0),
            valid=not payload.get("degraded", False),
        )

    # -- evaluación --

    def _prune(self, now: datetime) -> None:
        horizon = self._fusion_cfg.track_max_age_s * 4
        for ref in [
            r
            for r, t in self._tracks.items()
            if (now - t.observed_at).total_seconds() > horizon
        ]:
            del self._tracks[ref]
            self._identities.pop(ref, None)
            self._signatures.pop(ref, None)

    def evaluate(self, now: datetime | None = None) -> list[DecisionRecord]:
        """Decisiones nuevas o cambiadas desde la última evaluación."""
        now = now or datetime.now(UTC)
        self._prune(now)
        records = self._engine.evaluate(
            FusionInput(
                tracks=tuple(self._tracks.values()),
                identities=tuple(self._identities.values()),
                locations=tuple(self._locations.values()),
            ),
            now,
        )
        changed: list[DecisionRecord] = []
        for record in records:
            key = record.track_ref or f"tag:{record.person_id}"
            signature = (
                record.outcome,
                record.reason_codes,
                record.person_id,
                record.zone_id,
            )
            if self._signatures.get(key) != signature:
                self._signatures[key] = signature
                changed.append(record)
        return changed

    async def _tick(self) -> None:
        changed = self.evaluate()
        if changed:
            await publish_decisions(self._hub, changed, source=_SOURCE)
            self.decisions += len(changed)
            self.published += len(changed)

    def health(self) -> dict[str, Any]:
        base = super().health()
        base.update(
            decisions=self.decisions,
            tracks=len(self._tracks),
            unknown_locations=self.unknown_locations,
        )
        return base
