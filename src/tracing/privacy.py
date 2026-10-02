"""Filtro de privacidad aplicado **antes** de enviar nada a un servicio externo.

Postura: *deny-by-default*. Solo sale lo que una regla nombra explícitamente;
todo lo demás se descarta y solo se **cuenta** (``redacted_fields``, sin
contenido). Reglas por tipo de campo:

- **Vocabularios cerrados** (``outcome``, ``reason_codes``, ``kind``, ``role`` de
  evidencia): los valores de ``fusion.decision``. Cualquier otro valor se descarta.
- **Etiquetas cortas** (``type``, ``action``, ``target``, ``state``, ``stage``,
  ``status``): ``^[a-z][a-z0-9_]{0,31}$``. Nada de cadenas largas, mayúsculas,
  espacios, ``@``, base64 o ``data:``.
- **Identificadores** (``event_id``/``correlation_id``/``causation_id``,
  ``decision_id``, ``evidence_id``, ``track_ref``, ``stream_id``, ``zone_id``,
  ``source``): pasan tal cual solo si tienen el formato que generan el bus y la
  fusión (hex + sufijos de derivación, ``dec-<n>``, etiquetas de cámara/zona);
  si no, se sustituyen por un HMAC. ``person_id`` **siempre** se envía como HMAC.
- **Números**: ``confidence`` finito en [0, 1]; NaN/inf se descartan.
- **Fechas**: ISO 8601 válidas, re-serializadas.
- Listas de objetos: solo ``detections`` y ``tracks``, y solo como conteo.

HMAC (``h-<16 hex>``): SHA-256 con una clave local (``PrivacyFilter(key)`` o la
variable ``TRACING_HASH_KEY``; sin clave, una aleatoria por proceso, así los
hashes no son estables entre reinicios). Con la misma clave, el mismo valor da
el mismo hash, de modo que las relaciones entre saltos se conservan en Langfuse.

Limitación: el proyecto aún no tiene un seudonimizador propio, así que no existe
un "formato de seudónimo" que reconocer; por eso ``person_id`` nunca sale en
claro aunque *parezca* un seudónimo. Un formato validado no prueba
seudonimia, un HMAC con clave local sí impide revertir el valor fuera del sitio.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from bus import Topic
from fusion.decision import DecisionOutcome, EvidenceKind, EvidenceRole, ReasonCode

ENV_HASH_KEY = "TRACING_HASH_KEY"

_SHORT_LABEL = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_ERROR_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_PLAIN_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")  # cámara, zona, rol
_SYSTEM_ID = re.compile(
    r"^(?:[0-9a-f]{8,64}|restart|dec-[0-9]{1,8})(?:/[A-Za-z0-9_-]{1,32}){0,4}$"
)

_OUTCOMES = frozenset(o.value for o in DecisionOutcome)
_REASON_CODES = frozenset(c.value for c in ReasonCode)
_EVIDENCE_KINDS = frozenset(k.value for k in EvidenceKind)
_EVIDENCE_ROLES = frozenset(r.value for r in EvidenceRole)
_TOPICS = frozenset(t.value for t in Topic)

_SHORT_LABEL_FIELDS = ("type", "action", "target", "state", "stage", "status")
_ID_FIELDS = ("decision_id", "track_ref")
_PLAIN_FIELDS = ("stream_id", "zone_id")
_DATE_FIELDS = ("evaluated_at",)
_COUNTED_LISTS = ("detections", "tracks")
_MAX_LIST_ITEMS = 32
_ENVELOPE_ID_FIELDS = ("event_id", "correlation_id", "causation_id")

_NOT_VALID = object()


def _confidence(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _NOT_VALID
    if not math.isfinite(value) or not 0 <= value <= 1:
        return _NOT_VALID
    return value


def _iso(value: Any) -> Any:
    if not isinstance(value, str):
        return _NOT_VALID
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return _NOT_VALID


def _latency(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


class PrivacyFilter:
    """Allowlist + HMAC con contador de campos descartados."""

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key or os.urandom(32)
        self.redacted_fields = 0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PrivacyFilter:
        env = os.environ if env is None else env
        raw = env.get(ENV_HASH_KEY, "").strip()
        return cls(raw.encode("utf-8") if raw else None)

    # -- primitivas --

    def hash(self, value: object) -> str:
        digest = hmac.new(self._key, str(value).encode("utf-8", "replace"), hashlib.sha256)
        return "h-" + digest.hexdigest()[:16]

    def _system_id(self, value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        return value if _SYSTEM_ID.match(value) else self.hash(value)

    def _plain_label(self, value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        return value if _PLAIN_LABEL.match(value) else self.hash(value)

    # -- payload --

    def _evidence(self, items: Any) -> tuple[list[dict[str, Any]], int]:
        dropped = 0
        out: list[dict[str, Any]] = []
        if not isinstance(items, list):
            return out, 1
        for item in items[:_MAX_LIST_ITEMS]:
            if not isinstance(item, Mapping):
                dropped += 1
                continue
            clean: dict[str, Any] = {}
            for key, value in item.items():
                ok: Any = None
                if key == "evidence_id":
                    ok = self._system_id(value)
                elif key == "kind":
                    ok = value if value in _EVIDENCE_KINDS else None
                elif key == "role":
                    ok = value if value in _EVIDENCE_ROLES else None
                elif key == "confidence":
                    ok = None if (c := _confidence(value)) is _NOT_VALID else c
                elif key == "observed_at":
                    ok = None if (d := _iso(value)) is _NOT_VALID else d
                elif key in _PLAIN_FIELDS:
                    ok = self._plain_label(value)
                if ok is None:
                    dropped += 1
                else:
                    clean[key] = ok
            if "evidence_id" in clean:
                out.append(dict(sorted(clean.items())))
        return out, dropped

    def payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        dropped = 0
        for key, value in payload.items():
            clean: Any = None
            if key == "person_id":
                clean = self.hash(value)
            elif key == "outcome":
                clean = value if isinstance(value, str) and value in _OUTCOMES else None
            elif key == "reason_codes":
                if isinstance(value, list):
                    codes = [v for v in value[:_MAX_LIST_ITEMS] if isinstance(v, str) and v in _REASON_CODES]
                    dropped += len(value) - len(codes)
                    if codes:
                        out[key] = codes
                    continue
            elif key == "confidence":
                clean = None if (c := _confidence(value)) is _NOT_VALID else c
            elif key == "requires_operator":
                clean = value if isinstance(value, bool) else None
            elif key == "count":
                clean = value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**9 else None
            elif key in _SHORT_LABEL_FIELDS:
                clean = value if isinstance(value, str) and _SHORT_LABEL.match(value) else None
            elif key == "error_type":
                clean = value if isinstance(value, str) and _ERROR_TYPE.match(value) else None
            elif key in _ID_FIELDS:
                clean = self._system_id(value)
            elif key in _PLAIN_FIELDS:
                clean = self._plain_label(value)
            elif key in _DATE_FIELDS:
                clean = None if (d := _iso(value)) is _NOT_VALID else d
            elif key == "evidence":
                refs, bad = self._evidence(value)
                dropped += bad
                out[key] = refs
                continue
            elif key in _COUNTED_LISTS and isinstance(value, list) and all(
                isinstance(v, Mapping) for v in value
            ):
                out[f"{key}_count"] = len(value)
                continue
            if clean is None:
                dropped += 1
            else:
                out[key] = clean
        if dropped:
            self.redacted_fields += dropped
            out["redacted_fields"] = dropped
        return out

    # -- envelope y saltos --

    def envelope(self, data: Mapping[str, Any]) -> dict[str, Any]:
        topic = data.get("topic")
        created = _iso(data.get("created_at"))
        out: dict[str, Any] = {
            "topic": topic if topic in _TOPICS else None,
            "source": self._plain_label(data.get("source")),
            "stream_id": self._plain_label(data.get("stream_id")),
            "created_at": None if created is _NOT_VALID else created,
        }
        for key in _ENVELOPE_ID_FIELDS:
            out[key] = self._system_id(data.get(key))
        out["payload"] = self.payload(data.get("payload") or {})
        return out

    def hop(self, hop: Mapping[str, Any]) -> dict[str, Any]:
        """Un salto de ``TraceStore``: solo campos de trazabilidad, ids validados."""
        created = _iso(hop.get("created_at"))
        topic = hop.get("topic")
        parents = hop.get("parent_event_ids")
        return {
            "event_id": self._system_id(hop.get("event_id")),
            "topic": topic if topic in _TOPICS else None,
            "source": self._plain_label(hop.get("source")),
            "stream_id": self._plain_label(hop.get("stream_id")),
            "causation_id": self._system_id(hop.get("causation_id")),
            "parent_event_ids": [
                p for p in (self._system_id(x) for x in (parents or [])[:_MAX_LIST_ITEMS]) if p
            ],
            "created_at": None if created is _NOT_VALID else created,
            "hop_latency_ms": _latency(hop.get("hop_latency_ms")),
        }


_DEFAULT = PrivacyFilter()


def sanitize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atajo con un filtro de clave aleatoria por proceso."""
    return _DEFAULT.payload(payload)


def sanitize_envelope(data: Mapping[str, Any]) -> dict[str, Any]:
    return _DEFAULT.envelope(data)
