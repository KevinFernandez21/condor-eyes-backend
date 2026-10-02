"""Motor de fusión: puro, determinista y conservador.

Reglas de diseño:
- No lee el reloj: el instante de evaluación `now` es un parámetro.
- Ordena la evidencia por (marca de tiempo, evidence_id), así que el orden de llegada
  no cambia el resultado.
- Evidencia ausente, caducada, inválida o de baja confianza nunca cuenta a favor:
  solo `CORROBORATED` exige rostro, tag, zona, permiso y confianza, todo vigente.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from .config import FusionConfig
from .decision import (
    ALERT_CODES,
    DecisionOutcome,
    DecisionRecord,
    EvidenceKind,
    EvidenceRef,
    EvidenceRole,
    ReasonCode,
)
from .models import (
    FusionInput,
    IdentityEvidence,
    IdentityStatus,
    LocationEvidence,
    PermissionRepository,
    ReidLink,
    ReidStatus,
    TrackObservation,
    ZoneRule,
)

_CODE_ORDER = {code: i for i, code in enumerate(ReasonCode)}


class _Fresh(StrEnum):
    OK = "ok"
    STALE = "stale"
    FUTURE = "future"


class _Loc(StrEnum):
    OK = "ok"
    OTHER_ZONE = "other_zone"
    STALE = "stale"
    INVALID = "invalid"
    FUTURE = "future"
    LOW = "low"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class _Resolved:
    """Identidad resuelta para una pista y la evidencia que la sustenta."""

    person_id: str | None
    confidence: float | None
    codes: tuple[ReasonCode, ...]
    refs: tuple[EvidenceRef, ...]


def _key(
    item: IdentityEvidence | LocationEvidence | ReidLink | TrackObservation,
) -> tuple[datetime, str]:
    return (item.observed_at, item.evidence_id)


class FusionEngine:
    """Correlaciona pistas, identidad, ubicación de tags, permisos y reglas de zona."""

    def __init__(
        self,
        config: FusionConfig,
        zones: Iterable[ZoneRule],
        permissions: PermissionRepository,
    ) -> None:
        self._cfg = config
        self._zones = {z.zone_id: z for z in zones}
        self._permissions = permissions

    # ------------------------------------------------------------------ API

    def evaluate(
        self, evidence: FusionInput, now: datetime
    ) -> tuple[DecisionRecord, ...]:
        """Una decisión por pista, más una por cada tag sin persona visible."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now debe incluir zona horaria")
        all_tracks = sorted(evidence.tracks, key=_key)
        identities = sorted(evidence.identities, key=_key)
        reids = sorted(evidence.reids, key=_key)
        locations = sorted(evidence.locations, key=_key)

        # Una pista por track_ref: se conserva la más reciente. Si las observaciones
        # discrepan en zona o cámara, no se decide nada con ellas (DUPLICATE_TRACK).
        by_ref: dict[str, list[TrackObservation]] = {}
        for t in all_tracks:
            by_ref.setdefault(t.track_ref, []).append(t)
        conflicts = {
            ref: group
            for ref, group in by_ref.items()
            if len({(t.zone_id, t.stream_id) for t in group}) > 1
        }
        tracks = [g[-1] for ref, g in by_ref.items() if ref not in conflicts]
        tracks.sort(key=_key)

        direct = {
            t.track_ref: self._direct_identity(t, identities, now) for t in tracks
        }
        resolved = {
            t.track_ref: self._with_reid(t, direct, tracks, reids, now) for t in tracks
        }
        holders: dict[str, int] = {}
        for t in tracks:
            person = direct[t.track_ref].person_id
            if person is not None:
                holders[person] = holders.get(person, 0) + 1
        duplicated = {p for p, n in holders.items() if n > 1}

        records = [
            self._decide_track(t, resolved, tracks, locations, duplicated, now)
            for t in tracks
        ]
        for group in conflicts.values():
            refs = tuple(
                EvidenceRef(
                    t.evidence_id,
                    EvidenceKind.TRACK,
                    EvidenceRole.CONFLICTS,
                    t.observed_at,
                    t.confidence,
                    t.stream_id,
                    t.zone_id,
                    "observaciones discrepantes",
                )
                for t in group
            )
            records.append(
                self._record(
                    now, group[-1], None, (ReasonCode.DUPLICATE_TRACK,), refs, None
                )
            )
        records.extend(self._orphan_tags(tracks, resolved, locations, now))
        return tuple(records)

    # ------------------------------------------------------------ frescura

    def _fresh(self, at: datetime, max_age_s: float, now: datetime) -> _Fresh:
        age = (now - at).total_seconds()
        if age < -self._cfg.max_clock_skew_s:
            return _Fresh.FUTURE
        if age > max_age_s:
            return _Fresh.STALE
        return _Fresh.OK

    def _paired(self, at: datetime, track: TrackObservation) -> bool:
        return abs((at - track.observed_at).total_seconds()) <= self._cfg.time_window_s

    # ------------------------------------------------------------ identidad

    def _direct_identity(
        self,
        track: TrackObservation,
        identities: Sequence[IdentityEvidence],
        now: datetime,
    ) -> _Resolved:
        candidates = [i for i in identities if i.track_ref == track.track_ref]
        if not candidates:
            return _Resolved(None, None, (ReasonCode.IDENTITY_MISSING,), ())
        ev = candidates[-1]
        fresh = self._fresh(ev.observed_at, self._cfg.identity_max_age_s, now)
        if fresh is _Fresh.OK and not self._paired(ev.observed_at, track):
            fresh = _Fresh.STALE

        def ref(role: EvidenceRole, conf: float | None, detail: str) -> EvidenceRef:
            return EvidenceRef(
                ev.evidence_id,
                EvidenceKind.IDENTITY,
                role,
                ev.observed_at,
                conf,
                track.stream_id,
                track.zone_id,
                detail,
            )

        if fresh is _Fresh.FUTURE:
            return _Resolved(
                None,
                None,
                (ReasonCode.CLOCK_SKEW,),
                (ref(EvidenceRole.STALE, None, "futura"),),
            )
        if fresh is _Fresh.STALE:
            return _Resolved(
                None,
                None,
                (ReasonCode.STALE_IDENTITY,),
                (ref(EvidenceRole.STALE, None, "caducada"),),
            )
        status = ev.status
        if status is IdentityStatus.MATCH and ev.person_id:
            if ev.score >= self._cfg.min_identity_score:
                return _Resolved(
                    ev.person_id,
                    ev.score,
                    (),
                    (ref(EvidenceRole.SUPPORTS, ev.score, "match"),),
                )
            return _Resolved(
                None,
                None,
                (ReasonCode.LOW_CONFIDENCE,),
                (ref(EvidenceRole.CONTEXT, None, "score bajo"),),
            )
        if status is IdentityStatus.UNKNOWN:
            return _Resolved(
                None,
                None,
                (ReasonCode.UNKNOWN_FACE,),
                (ref(EvidenceRole.CONFLICTS, None, "desconocido"),),
            )
        if status is IdentityStatus.NO_FACE:
            return _Resolved(
                None,
                None,
                (ReasonCode.HIDDEN_FACE,),
                (ref(EvidenceRole.CONTEXT, None, "sin rostro"),),
            )
        return _Resolved(
            None,
            None,
            (ReasonCode.FACE_INCONCLUSIVE,),
            (ref(EvidenceRole.CONTEXT, None, status.value),),
        )

    def _with_reid(
        self,
        track: TrackObservation,
        direct: dict[str, _Resolved],
        tracks: Sequence[TrackObservation],
        reids: Sequence[ReidLink],
        now: datetime,
    ) -> _Resolved:
        own = direct[track.track_ref]
        if own.person_id is not None or ReasonCode.UNKNOWN_FACE in own.codes:
            return own
        for link in reversed(reids):
            if link.status is not ReidStatus.LINKED or link.candidate_track_ref is None:
                continue
            if track.track_ref == link.source_track_ref:
                other = link.candidate_track_ref
            elif track.track_ref == link.candidate_track_ref:
                other = link.source_track_ref
            else:
                continue
            other_res = direct.get(other)
            if (
                other_res is None
                or other_res.person_id is None
                or other_res.confidence is None
            ):
                continue
            if link.confidence < self._cfg.min_reid_confidence:
                continue
            if (
                self._fresh(link.observed_at, self._cfg.reid_max_age_s, now)
                is not _Fresh.OK
            ):
                continue
            conf = min(link.confidence, other_res.confidence)
            reid_ref = EvidenceRef(
                link.evidence_id,
                EvidenceKind.REID,
                EvidenceRole.SUPPORTS,
                link.observed_at,
                link.confidence,
                track.stream_id,
                track.zone_id,
                f"enlazada con {other}",
            )
            return _Resolved(other_res.person_id, conf, (), (*other_res.refs, reid_ref))
        return own

    # ------------------------------------------------------------ ubicación

    def _classify_location(
        self,
        loc: LocationEvidence,
        track: TrackObservation | None,
        now: datetime,
        zone: str,
    ) -> _Loc:
        if not loc.valid:
            return _Loc.INVALID
        fresh = self._fresh(loc.observed_at, self._cfg.location_max_age_s, now)
        if fresh is _Fresh.FUTURE:
            return _Loc.FUTURE
        if fresh is _Fresh.STALE or (
            track is not None and not self._paired(loc.observed_at, track)
        ):
            return _Loc.STALE
        if loc.confidence < self._cfg.min_location_confidence:
            return _Loc.LOW
        return _Loc.OK if loc.zone_id == zone else _Loc.OTHER_ZONE

    def _person_location(
        self,
        person: str,
        locations: Sequence[LocationEvidence],
        track: TrackObservation | None,
        now: datetime,
        zone: str,
    ) -> tuple[_Loc, LocationEvidence | None]:
        """Se juzga la lectura más reciente: una posterior inválida, de baja confianza,
        futura o de otra zona degrada a la anterior, que no puede corroborar sola."""
        own = [l for l in locations if l.person_id == person]
        if not own:
            return _Loc.MISSING, None
        latest = own[-1]
        return self._classify_location(latest, track, now, zone), latest

    @staticmethod
    def _loc_ref(loc: LocationEvidence, role: EvidenceRole, detail: str) -> EvidenceRef:
        return EvidenceRef(
            loc.evidence_id,
            EvidenceKind.LOCATION,
            role,
            loc.observed_at,
            loc.confidence,
            None,
            loc.zone_id,
            detail,
        )

    # ------------------------------------------------------------ decisión

    def _decide_track(
        self,
        track: TrackObservation,
        resolved: dict[str, _Resolved],
        tracks: Sequence[TrackObservation],
        locations: Sequence[LocationEvidence],
        duplicated: set[str],
        now: datetime,
    ) -> DecisionRecord:
        def track_ref(role: EvidenceRole, detail: str = "") -> EvidenceRef:
            return EvidenceRef(
                track.evidence_id,
                EvidenceKind.TRACK,
                role,
                track.observed_at,
                track.confidence,
                track.stream_id,
                track.zone_id,
                detail,
            )

        def finish(
            codes: Sequence[ReasonCode],
            refs: Sequence[EvidenceRef],
            person: str | None,
            outcome: DecisionOutcome | None = None,
        ) -> DecisionRecord:
            return self._record(now, track, person, codes, refs, outcome)

        rule = self._zones.get(track.zone_id)
        if rule is None:
            return finish(
                (ReasonCode.ZONE_UNKNOWN,), (track_ref(EvidenceRole.CONTEXT),), None
            )
        if not rule.restricted:
            return finish(
                (ReasonCode.ZONE_UNRESTRICTED,),
                (track_ref(EvidenceRole.CONTEXT),),
                None,
                DecisionOutcome.NOT_RESTRICTED,
            )
        fresh = self._fresh(track.observed_at, self._cfg.track_max_age_s, now)
        if fresh is _Fresh.FUTURE:
            return finish(
                (ReasonCode.CLOCK_SKEW,),
                (track_ref(EvidenceRole.STALE, "futura"),),
                None,
            )
        if fresh is _Fresh.STALE:
            return finish(
                (ReasonCode.STALE_VISION,),
                (track_ref(EvidenceRole.STALE, "caducada"),),
                None,
            )
        if track.confidence < self._cfg.min_track_confidence:
            return finish(
                (ReasonCode.LOW_CONFIDENCE,),
                (track_ref(EvidenceRole.CONTEXT, "confianza baja"),),
                None,
            )

        ident = resolved[track.track_ref]
        refs: list[EvidenceRef] = [track_ref(EvidenceRole.SUPPORTS), *ident.refs]
        codes: list[ReasonCode] = list(ident.codes)
        person = ident.person_id

        if person is not None:
            if person in duplicated:
                codes.append(ReasonCode.DUPLICATE_IDENTITY)
            allowed = self._permissions.allowed_zones(person)
            if allowed is None:
                codes.append(ReasonCode.NO_PERMISSION_RECORD)
            elif track.zone_id not in allowed:
                codes.append(ReasonCode.ZONE_NOT_PERMITTED)
            perm_role = (
                EvidenceRole.SUPPORTS
                if allowed and track.zone_id in allowed
                else EvidenceRole.CONFLICTS
            )
            refs.append(
                EvidenceRef(
                    f"permission:{person}",
                    EvidenceKind.PERMISSION,
                    perm_role if allowed is not None else EvidenceRole.CONTEXT,
                    None,
                    None,
                    None,
                    track.zone_id,
                    "sin registro"
                    if allowed is None
                    else "zona permitida"
                    if track.zone_id in allowed
                    else "zona no permitida",
                )
            )
            state, loc = self._person_location(
                person, locations, track, now, track.zone_id
            )
            codes.extend(self._location_codes(state))
            if loc is not None:
                role = (
                    EvidenceRole.SUPPORTS
                    if state is _Loc.OK
                    else EvidenceRole.CONFLICTS
                    if state is _Loc.OTHER_ZONE
                    else EvidenceRole.STALE
                )
                refs.append(self._loc_ref(loc, role, state.value))
        else:
            codes.extend(
                self._tag_context(track, resolved, tracks, locations, now, refs)
            )

        return finish(codes, refs, person)

    @staticmethod
    def _location_codes(state: _Loc) -> list[ReasonCode]:
        return {
            _Loc.OK: [],
            _Loc.OTHER_ZONE: [ReasonCode.IDENTITY_TAG_MISMATCH],
            _Loc.STALE: [ReasonCode.STALE_SENSOR],
            _Loc.INVALID: [ReasonCode.LOCATION_INVALID],
            _Loc.FUTURE: [ReasonCode.CLOCK_SKEW],
            _Loc.LOW: [ReasonCode.LOW_CONFIDENCE],
            _Loc.MISSING: [ReasonCode.PERSON_WITHOUT_TAG],
        }[state]

    def _tag_context(
        self,
        track: TrackObservation,
        resolved: dict[str, _Resolved],
        tracks: Sequence[TrackObservation],
        locations: Sequence[LocationEvidence],
        now: datetime,
        refs: list[EvidenceRef],
    ) -> list[ReasonCode]:
        """Sin identidad no se puede atribuir un tag a la persona: solo se explica el contexto."""
        zone = track.zone_id
        in_zone_ok: list[LocationEvidence] = []
        problems: dict[_Loc, LocationEvidence] = {}
        for person in sorted({l.person_id for l in locations}):
            state, loc = self._person_location(person, locations, track, now, zone)
            if loc is None or loc.zone_id != zone:
                continue
            if state is _Loc.OK:
                in_zone_ok.append(loc)
                refs.append(self._loc_ref(loc, EvidenceRole.CONTEXT, "tag en la zona"))
            else:
                problems.setdefault(state, loc)
                refs.append(self._loc_ref(loc, EvidenceRole.STALE, state.value))
        codes: list[ReasonCode] = []
        if not in_zone_ok:
            explained = False
            for state in (_Loc.STALE, _Loc.INVALID, _Loc.FUTURE, _Loc.LOW):
                if state in problems:
                    codes.extend(self._location_codes(state))
                    explained = True
            if not explained:
                codes.append(ReasonCode.PERSON_WITHOUT_TAG)
        n_people = len(self._fresh_tracks_in_zone(tracks, zone, now))
        if n_people > 1 or len(in_zone_ok) > 1:
            codes.append(ReasonCode.MULTIPLE_NEARBY_PEOPLE)
        return codes

    def _fresh_tracks_in_zone(
        self, tracks: Sequence[TrackObservation], zone: str, now: datetime
    ) -> list[TrackObservation]:
        return [
            t
            for t in tracks
            if t.zone_id == zone
            and self._fresh(t.observed_at, self._cfg.track_max_age_s, now) is _Fresh.OK
        ]

    def _orphan_tags(
        self,
        tracks: Sequence[TrackObservation],
        resolved: dict[str, _Resolved],
        locations: Sequence[LocationEvidence],
        now: datetime,
    ) -> list[DecisionRecord]:
        records = []
        for person in sorted({l.person_id for l in locations}):
            own = [l for l in locations if l.person_id == person]
            loc = own[-1]  # solo cuenta la lectura más reciente
            rule = self._zones.get(loc.zone_id)
            if rule is None or not rule.restricted:
                continue
            if self._classify_location(loc, None, now, loc.zone_id) is not _Loc.OK:
                continue
            present = [
                t
                for t in self._fresh_tracks_in_zone(tracks, loc.zone_id, now)
                if resolved[t.track_ref].person_id in (None, person)
            ]
            if present:
                continue
            refs = (
                self._loc_ref(loc, EvidenceRole.SUPPORTS, "tag sin persona visible"),
            )
            records.append(
                self._record(
                    now,
                    None,
                    person,
                    (ReasonCode.TAG_WITHOUT_PERSON,),
                    refs,
                    None,
                    zone=loc.zone_id,
                )
            )
        return records

    # ------------------------------------------------------------ registro

    def _record(
        self,
        now: datetime,
        track: TrackObservation | None,
        person: str | None,
        codes: Sequence[ReasonCode],
        refs: Sequence[EvidenceRef],
        forced: DecisionOutcome | None,
        zone: str | None = None,
    ) -> DecisionRecord:
        unique_refs: dict[tuple[str, EvidenceRole], EvidenceRef] = {}
        for r in refs:
            unique_refs.setdefault((r.evidence_id, r.role), r)
        evidence = tuple(unique_refs.values())
        confs = [
            r.confidence
            for r in evidence
            if r.confidence is not None
            and r.role in (EvidenceRole.SUPPORTS, EvidenceRole.CONFLICTS)
        ]
        confidence = min(confs) if confs else 0.0

        code_set = set(codes)
        if forced is not None:
            outcome = forced
        elif not code_set:
            if confidence >= self._cfg.min_decision_confidence:
                outcome = DecisionOutcome.CORROBORATED
                code_set = {ReasonCode.EVIDENCE_CONSISTENT}
            else:
                outcome = DecisionOutcome.INCONCLUSIVE
                code_set = {ReasonCode.LOW_CONFIDENCE}
        elif code_set & ALERT_CODES:
            outcome = DecisionOutcome.ALERT
        else:
            outcome = DecisionOutcome.INCONCLUSIVE

        ordered = tuple(sorted(code_set, key=_CODE_ORDER.__getitem__))
        material = "|".join(
            [
                track.track_ref if track else "",
                person or "",
                zone or (track.zone_id if track else ""),
                now.isoformat(),
                *sorted(
                    f"{r.evidence_id}@{r.observed_at.isoformat() if r.observed_at else ''}:{r.role.value}"
                    for r in evidence
                ),
            ]
        )
        decision_id = hashlib.sha256(material.encode()).hexdigest()[:24]
        return DecisionRecord(
            decision_id=decision_id,
            evaluated_at=now,
            outcome=outcome,
            confidence=confidence,
            reason_codes=ordered,
            evidence=evidence,
            track_ref=track.track_ref if track else None,
            stream_id=track.stream_id if track else None,
            zone_id=zone or (track.zone_id if track else None),
            person_id=person,
        )
