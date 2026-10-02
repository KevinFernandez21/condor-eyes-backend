"""Filtro de privacidad aplicado **antes** de enviar nada a un servicio externo.

Es una allowlist: lo que no está nombrado aquí no sale. Reglas:

- Solo campos explícitamente permitidos; el resto se descarta (embeddings,
  imágenes, recortes de rostro, IDs de tag crudos, nombres, cajas, etc.).
- Los identificadores de persona solo salen si tienen forma de seudónimo
  (``p-<hex>``, ``anon-<hex>``); cualquier otro valor se sustituye por
  ``[redacted]``. Los IDs de tag crudos no están en la allowlist.
- Las listas de objetos (detecciones, tracks) se reducen a ``<campo>_count``;
  solo ``evidence`` conserva sus referencias, y únicamente con campos permitidos.
- Los valores permitidos se vuelven a comprobar: nada de estructuras en campos
  escalares, ``data:`` URIs ni cadenas largas.

El filtro es defensa en profundidad: el bus ya rechaza binarios y arrays.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[redacted]"
MAX_STRING = 120

_PSEUDONYM = re.compile(r"^(?:p|anon|ps)-[0-9a-f]{4,64}$")

_SCALAR_FIELDS = frozenset(
    {
        "decision_id", "evaluated_at", "outcome", "confidence", "track_ref", "stream_id",
        "zone_id", "requires_operator", "type", "action", "target", "state", "role",
        "stage", "error_type", "count", "kind", "status",
    }
)
_IDENTITY_FIELDS = frozenset({"person_id"})
_SCALAR_LIST_FIELDS = frozenset({"reason_codes"})
_EVIDENCE_FIELDS = frozenset(
    {"evidence_id", "kind", "role", "observed_at", "confidence", "stream_id", "zone_id"}
)
_ENVELOPE_FIELDS = (
    "topic", "source", "stream_id", "created_at", "event_id", "correlation_id", "causation_id",
)
_MAX_LIST_ITEMS = 32


def is_pseudonym(value: object) -> bool:
    """True si ``value`` tiene la forma de un seudónimo hash (``p-7f3a``)."""
    return isinstance(value, str) and _PSEUDONYM.match(value) is not None


def _safe_scalar(value: Any) -> tuple[bool, Any]:
    """Devuelve (ok, valor) para un escalar JSON inocuo; si no, (False, None)."""
    if value is None or isinstance(value, bool | int | float):
        return True, value
    if isinstance(value, str):
        if value.startswith("data:"):
            return False, None
        return True, value[:MAX_STRING]
    return False, None


def _sanitize_evidence(items: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for item in items[:_MAX_LIST_ITEMS]:
        if not isinstance(item, Mapping):
            continue
        clean: dict[str, Any] = {}
        for key in _EVIDENCE_FIELDS:
            if key in item:
                ok, value = _safe_scalar(item[key])
                if ok:
                    clean[key] = value
        if clean:
            out.append(dict(sorted(clean.items())))
    return out


def sanitize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Devuelve una copia del payload con solo lo permitido."""
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _IDENTITY_FIELDS:
            out[key] = value if is_pseudonym(value) else REDACTED
        elif key in _SCALAR_FIELDS:
            ok, safe = _safe_scalar(value)
            if ok:
                out[key] = safe
        elif key in _SCALAR_LIST_FIELDS:
            if isinstance(value, list):
                items = [_safe_scalar(v) for v in value[:_MAX_LIST_ITEMS]]
                out[key] = [v for ok, v in items if ok]
        elif key == "evidence":
            out[key] = _sanitize_evidence(value)
        elif isinstance(value, list) and value and all(isinstance(v, Mapping) for v in value):
            out[f"{key}_count"] = len(value)
    return out


def sanitize_envelope(data: Mapping[str, Any]) -> dict[str, Any]:
    """Envelope serializado reducido a campos de trazabilidad y payload filtrado."""
    out: dict[str, Any] = {key: data.get(key) for key in _ENVELOPE_FIELDS}
    out["payload"] = sanitize_payload(data.get("payload") or {})
    return out
