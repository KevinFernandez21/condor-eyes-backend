"""Filtro de privacidad aplicado **antes** de enviar nada a un servicio externo.

Postura: *deny-by-default* y **sin reconocer ids "seguros" de forma libre**. Solo
sale lo que una regla nombra; lo demás se descarta y solo se **cuenta**
(``redacted_fields``, nunca el contenido). El filtro es *total*: ante cualquier
tipo inesperado descarta y cuenta, no lanza.

- **Identificadores** (``event_id``, ``correlation_id``, ``causation_id``,
  ``decision_id``, ``evidence_id``, ``track_ref``, ``person_id`` y los padres de
  un salto): **siempre** HMAC ``h-<16 hex>``. El mismo valor da el mismo hash con
  la misma clave, así que el enlace entre saltos se conserva en Langfuse; el id
  crudo vive solo en el ``TraceStore`` local.
- **stream_id / zone_id / source**: HMAC salvo que el valor esté en una allowlist
  explícita (constructor o ``TRACING_ALLOWED_STREAMS`` / ``TRACING_ALLOWED_ZONES``
  / ``TRACING_ALLOWED_SOURCES``). Por defecto solo pasan los roles de agente.
- **Etiquetas** (``type``, ``action``, ``target``, ``role``, ``state``, ``stage``,
  ``status``, ``error_type``, ``outcome``, ``reason_codes``, ``kind``): vocabulario
  cerrado por campo, construido con los valores que el proyecto emite. Fuera del
  vocabulario: se descarta.
- ``confidence``: número finito en [0, 1]. Fechas: ISO 8601 válida, sin espacios.
- Listas de objetos: solo ``detections`` y ``tracks``, y solo como conteo.

HMAC = SHA-256 con clave local (``PrivacyFilter(key)`` o ``TRACING_HASH_KEY``,
mínimo 16 bytes); sin clave, una aleatoria por proceso (los hashes no son
estables entre reinicios). Un hash no es anonimato frente a quien tenga la clave.
"""

from __future__ import annotations

import builtins
import hashlib
import hmac
import math
import os
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from agents.runtime import WorkerState
from bus import Topic
from bus.hub import (
    HubError,
    InvalidEnvelopeError,
    InvalidTopicError,
    UnsupportedVersionError,
)
from events.model import EventType
from fusion.decision import DecisionOutcome, EvidenceKind, EvidenceRole, ReasonCode
from fusion.models import IdentityStatus, ReidStatus
from pipeline.health import StreamState

ENV_HASH_KEY = "TRACING_HASH_KEY"
ENV_ALLOWED_STREAMS = "TRACING_ALLOWED_STREAMS"
ENV_ALLOWED_ZONES = "TRACING_ALLOWED_ZONES"
ENV_ALLOWED_SOURCES = "TRACING_ALLOWED_SOURCES"
MIN_KEY_BYTES = 16

AGENT_ROLES: tuple[str, ...] = (
    "ingest", "inference", "tracker", "event", "storage", "comms", "supervisor",
)
"""Roles de ``agents.route.MULTIAGENT_ROUTE`` (un test comprueba que coinciden)."""

_DEFAULT_SOURCES = frozenset((*AGENT_ROLES, "fusion", "runtime", "hub"))


def _values(*enums: type) -> frozenset[str]:
    return frozenset(str(member.value) for enum in enums for member in enum)  # type: ignore[attr-defined]


_EXCEPTION_NAMES = frozenset(
    name
    for name, obj in vars(builtins).items()
    if isinstance(obj, type) and issubclass(obj, BaseException)
) | {c.__name__ for c in (HubError, InvalidEnvelopeError, InvalidTopicError, UnsupportedVersionError)}

_VOCAB: dict[str, frozenset[str]] = {
    "type": _values(EventType),
    "action": frozenset({"restart"}),
    "target": frozenset(AGENT_ROLES),
    "role": frozenset(AGENT_ROLES),
    "state": _values(WorkerState, StreamState) | {"up", "down"},
    "stage": frozenset({"handle", "consume", "heartbeat", "queue_overflow"}),
    "status": _values(IdentityStatus, ReidStatus),
    "error_type": _EXCEPTION_NAMES,
    "outcome": _values(DecisionOutcome),
    "kind": _values(EvidenceKind),
}
_REASON_CODES = _values(ReasonCode)
_EVIDENCE_KINDS = _values(EvidenceKind)
_EVIDENCE_ROLES = _values(EvidenceRole)
_TOPICS = _values(Topic)

_ID_FIELDS = ("decision_id", "track_ref", "person_id")
_ENVELOPE_ID_FIELDS = ("event_id", "correlation_id", "causation_id")
_COUNTED_LISTS = ("detections", "tracks")
_MAX_LIST_ITEMS = 32


class PrivacyConfigError(ValueError):
    """Configuración inválida del filtro de privacidad."""


_NOT_VALID = object()


def _confidence(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _NOT_VALID
    if not math.isfinite(value) or not 0 <= value <= 1:
        return _NOT_VALID
    return value


def _iso(value: Any) -> Any:
    if not isinstance(value, str) or value != value.strip():
        return _NOT_VALID
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return _NOT_VALID


def _latency(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _csv(raw: str | None) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


class PrivacyFilter:
    """Allowlist + HMAC con contador de campos descartados. Nunca lanza al filtrar."""

    def __init__(
        self,
        key: bytes | None = None,
        *,
        allowed_streams: Iterable[str] = (),
        allowed_zones: Iterable[str] = (),
        allowed_sources: Iterable[str] | None = None,
    ) -> None:
        if key is not None and len(key) < MIN_KEY_BYTES:
            raise PrivacyConfigError(
                f"La clave HMAC del filtro de privacidad debe tener al menos {MIN_KEY_BYTES} bytes"
            )
        self._key = key or os.urandom(32)
        self._streams = frozenset(allowed_streams)
        self._zones = frozenset(allowed_zones)
        self._sources = (
            _DEFAULT_SOURCES if allowed_sources is None else _DEFAULT_SOURCES | frozenset(allowed_sources)
        )
        self.redacted_fields = 0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PrivacyFilter:
        env = os.environ if env is None else env
        raw = env.get(ENV_HASH_KEY, "").strip()
        key = raw.encode("utf-8") if raw else None
        if key is not None and len(key) < MIN_KEY_BYTES:
            raise PrivacyConfigError(
                f"{ENV_HASH_KEY} debe tener al menos {MIN_KEY_BYTES} bytes "
                "(o quítala para usar una clave aleatoria por proceso)"
            )
        return cls(
            key,
            allowed_streams=_csv(env.get(ENV_ALLOWED_STREAMS)),
            allowed_zones=_csv(env.get(ENV_ALLOWED_ZONES)),
            allowed_sources=_csv(env.get(ENV_ALLOWED_SOURCES)),
        )

    # -- primitivas --

    def hash(self, value: object) -> str:
        digest = hmac.new(self._key, str(value).encode("utf-8", "replace"), hashlib.sha256)
        return "h-" + digest.hexdigest()[:16]

    def _id(self, value: Any) -> str | None:
        return self.hash(value) if isinstance(value, str) and value else None

    def _label(self, value: Any, allowed: frozenset[str]) -> str | None:
        """Etiqueta con allowlist explícita; fuera de ella, HMAC."""
        if not isinstance(value, str) or not value:
            return None
        return value if value in allowed else self.hash(value)

    @staticmethod
    def _vocab(field: str, value: Any) -> str | None:
        return value if isinstance(value, str) and value in _VOCAB[field] else None

    # -- evidencia --

    def _evidence(self, items: Any) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(items, list):
            return [], 1
        out: list[dict[str, Any]] = []
        dropped = 0
        for item in items[:_MAX_LIST_ITEMS]:
            if not isinstance(item, Mapping):
                dropped += 1
                continue
            clean: dict[str, Any] = {}
            for key, value in item.items():
                ok: Any = None
                try:
                    if key == "evidence_id":
                        ok = self._id(value)
                    elif key == "kind":
                        ok = value if isinstance(value, str) and value in _EVIDENCE_KINDS else None
                    elif key == "role":
                        ok = value if isinstance(value, str) and value in _EVIDENCE_ROLES else None
                    elif key == "confidence":
                        ok = None if (c := _confidence(value)) is _NOT_VALID else c
                    elif key == "observed_at":
                        ok = None if (d := _iso(value)) is _NOT_VALID else d
                    elif key == "stream_id":
                        ok = self._label(value, self._streams)
                    elif key == "zone_id":
                        ok = self._label(value, self._zones)
                except Exception:  # noqa: BLE001 - el filtro es total: descarta y cuenta
                    ok = None
                if ok is None:
                    dropped += 1
                else:
                    clean[key] = ok
            if "evidence_id" in clean:
                out.append(dict(sorted(clean.items())))
        return out, dropped

    # -- payload --

    def _field(self, key: Any, value: Any) -> Any:
        """Valor limpio de un campo de payload; ``None`` si se descarta."""
        if key in _ID_FIELDS:
            return self._id(value)
        if key in _VOCAB:
            return self._vocab(str(key), value)
        if key == "confidence":
            c = _confidence(value)
            return None if c is _NOT_VALID else c
        if key == "requires_operator":
            return value if isinstance(value, bool) else None
        if key == "count":
            ok = isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**9
            return value if ok else None
        if key == "stream_id":
            return self._label(value, self._streams)
        if key == "zone_id":
            return self._label(value, self._zones)
        if key == "evaluated_at":
            d = _iso(value)
            return None if d is _NOT_VALID else d
        return None

    def payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            self.redacted_fields += 1
            return {"redacted_fields": 1}
        out: dict[str, Any] = {}
        dropped = 0
        for key, value in payload.items():
            try:
                if key == "reason_codes":
                    if isinstance(value, list):
                        codes = [
                            v for v in value[:_MAX_LIST_ITEMS] if isinstance(v, str) and v in _REASON_CODES
                        ]
                        dropped += len(value) - len(codes)
                        if codes:
                            out[key] = codes
                    else:
                        dropped += 1
                    continue
                if key == "evidence":
                    refs, bad = self._evidence(value)
                    dropped += bad
                    out[key] = refs
                    continue
                if (
                    key in _COUNTED_LISTS
                    and isinstance(value, list)
                    and all(isinstance(v, Mapping) for v in value)
                ):
                    out[f"{key}_count"] = len(value)
                    continue
                clean = self._field(key, value)
            except Exception:  # noqa: BLE001 - el filtro es total: descarta y cuenta
                clean = None
            if clean is None:
                dropped += 1
            else:
                out[str(key)] = clean
        if dropped:
            self.redacted_fields += dropped
            out["redacted_fields"] = dropped
        return out

    # -- envelope y saltos --

    def envelope(self, data: Any) -> dict[str, Any]:
        out: dict[str, Any] = {
            "topic": None, "source": None, "stream_id": None, "created_at": None,
            "event_id": None, "correlation_id": None, "causation_id": None, "payload": {},
        }
        if not isinstance(data, Mapping):
            self.redacted_fields += 1
            return out
        try:
            topic = data.get("topic")
            out["topic"] = topic if isinstance(topic, str) and topic in _TOPICS else None
            out["source"] = self._label(data.get("source"), self._sources)
            out["stream_id"] = self._label(data.get("stream_id"), self._streams)
            created = _iso(data.get("created_at"))
            out["created_at"] = None if created is _NOT_VALID else created
            for key in _ENVELOPE_ID_FIELDS:
                out[key] = self._id(data.get(key))
        except Exception:  # noqa: BLE001 - el filtro es total: descarta y cuenta
            self.redacted_fields += 1
        out["payload"] = self.payload(data.get("payload") or {})
        return out

    def hop(self, hop: Any) -> dict[str, Any]:
        """Un salto de ``TraceStore``: solo campos de trazabilidad, ids con HMAC."""
        out: dict[str, Any] = {
            "event_id": None, "topic": None, "source": None, "stream_id": None,
            "causation_id": None, "parent_event_ids": [], "created_at": None,
            "hop_latency_ms": None,
        }
        if not isinstance(hop, Mapping):
            self.redacted_fields += 1
            return out
        try:
            out["event_id"] = self._id(hop.get("event_id"))
            topic = hop.get("topic")
            out["topic"] = topic if isinstance(topic, str) and topic in _TOPICS else None
            out["source"] = self._label(hop.get("source"), self._sources)
            out["stream_id"] = self._label(hop.get("stream_id"), self._streams)
            out["causation_id"] = self._id(hop.get("causation_id"))
            parents = hop.get("parent_event_ids")
            if isinstance(parents, list | tuple):
                out["parent_event_ids"] = [
                    p for p in (self._id(x) for x in parents[:_MAX_LIST_ITEMS]) if p
                ]
            created = _iso(hop.get("created_at"))
            out["created_at"] = None if created is _NOT_VALID else created
            out["hop_latency_ms"] = _latency(hop.get("hop_latency_ms"))
        except Exception:  # noqa: BLE001 - el filtro es total: descarta y cuenta
            self.redacted_fields += 1
        return out
