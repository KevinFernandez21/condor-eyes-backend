"""Funciones puras que convierten las respuestas de la API en vistas del dashboard.

Sin E/S de red ni estado: todo se puede probar con diccionarios. La única
entrada externa es ``load_site_map`` (lee un TOML opcional) y ``default_routes``
(lee las definiciones estáticas de rutas de ``agents``; no arranca nada).
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REDACTED = "[omitido]"
HARDWARE_LABEL = "laptop"

# Orden de gravedad al agregar instancias de un mismo rol.
_STATE_RANK = {"ok": 0, "stopped": 1, "degraded": 2, "failed": 3}


# --------------------------------------------------------------------------
# Rutas (nodos y aristas del grafo)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """Qué consume y publica un rol (nombres de tópico como texto)."""

    name: str
    consumes: tuple[str, ...] = ()
    publishes: tuple[str, ...] = ()
    planned: bool = False


# Roles que aún no tienen definición en ``agents``: se dibujan en gris hasta que
# la API los reporte. Los tópicos ``location`` e ``identity`` todavía no existen
# en el bus de esta rama; la arista se muestra sin tasa.
PLANNED_ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec("location", publishes=("location",), planned=True),
    RouteSpec("identity", publishes=("identity",), planned=True),
    RouteSpec(
        "fusion",
        consumes=("location", "identity", "vision.tracks"),
        publishes=("events",),
        planned=True,
    ),
    RouteSpec("actuation", consumes=("events",), planned=True),
)


# Posición (columna, fila) de cada rol en el lienzo; los desconocidos van abajo.
LAYOUT: dict[str, tuple[int, int]] = {
    # fila 0: cadena principal; fila 1: supervisor y comms; fila 2: plugins laterales
    "ingest": (0, 0), "inference": (1, 0), "tracker": (2, 0), "event": (3, 0), "storage": (4, 0),
    "supervisor": (2, 1), "comms": (4, 1),
    "location": (0, 2), "identity": (1, 2), "fusion": (2, 2), "actuation": (3, 2),
}  # fmt: skip
CONTROL_TOPICS = frozenset({"system.health", "system.commands", "system.errors"})


def default_routes() -> list[RouteSpec]:
    """Las siete rutas de ``agents.route`` más los roles previstos."""
    from agents.route import MULTIAGENT_ROUTE  # estático; no arranca agentes

    routes = [
        RouteSpec(
            r.name,
            consumes=tuple(str(t.value) for t in r.consumes),
            publishes=tuple(str(t.value) for t in r.publishes),
        )
        for r in MULTIAGENT_ROUTE
    ]
    return routes + list(PLANNED_ROUTES)


def agent_state(agent: Mapping[str, Any]) -> str:
    """ok | degraded | failed | stopped a partir del estado del worker."""
    raw = str(agent.get("state", "")).lower().rsplit(".", 1)[-1]
    if raw == "failed":
        return "failed"
    if raw in ("starting", "stopping"):
        return "degraded"
    if raw in ("stopped", "created"):
        return "stopped"
    if raw == "running":
        if agent.get("failures") or agent.get("last_error"):
            return "degraded"
        return "ok"
    return "degraded"


def _max_int(values: Iterable[Any]) -> int | None:
    nums = [int(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return max(nums) if nums else None


_COMPONENT_CLASS = {
    "ok": "ok", "simulated": "ok",
    "starting": "degraded", "degraded": "degraded",
    "disabled": "stopped", "not_used": "stopped", "stopped": "stopped",
}  # fmt: skip


def components(agents: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Componentes del ejecutor (cámara, detector, fusión…), aparte del grafo."""
    out = []
    for item in agents:
        if item.get("role") != "component":
            continue
        state = str(item.get("state", "")).lower()
        out.append(
            {
                "name": str(item.get("name", "")).removeprefix("component/"),
                "state": state or "—",
                "state_class": _COMPONENT_CLASS.get(state, "unknown"),
                "detail": item.get("last_error") or None,
            }
        )
    return sorted(out, key=lambda c: c["name"])


def _edge_label(topic: str, stats: Mapping[str, Any] | None) -> str:
    if stats is None:
        return topic
    rate = stats.get("rate_per_s")
    drops = stats.get("drops", 0)
    parts = [topic]
    if rate is not None:
        parts.append(f"{rate:g}/s")
    if drops:
        parts.append(f"{drops} descartes")
    return " · ".join(parts)


def build_graph(
    agents: Sequence[Mapping[str, Any]],
    topics: Sequence[Mapping[str, Any]],
    routes: Sequence[RouteSpec] | None = None,
    node_rates: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Nodos = roles; aristas = tópicos entre publicador y consumidor.

    Los componentes (``role == "component"``) no son nodos: van en ``components``.
    """
    routes = list(default_routes() if routes is None else routes)
    node_rates = node_rates or {}
    by_role: dict[str, list[Mapping[str, Any]]] = {}
    for item in agents:
        if item.get("role") == "component":
            continue
        by_role.setdefault(str(item.get("role", item.get("name", "?"))), []).append(item)
    known = {r.name for r in routes}
    for role in by_role:  # roles que la API reporta y no conocemos
        if role not in known:
            routes.append(RouteSpec(role))

    stats = {str(t.get("topic")): t for t in topics}
    nodes: list[dict[str, Any]] = []
    extra = 0
    for route in routes:
        if route.name in LAYOUT:
            col, row = LAYOUT[route.name]
        else:  # rol que no conocemos: fila nueva debajo, de a 5 por fila
            col, row = extra % 5, 3 + extra // 5
            extra += 1
        members = by_role.get(route.name, [])
        if members:
            state = max((agent_state(m) for m in members), key=_STATE_RANK.__getitem__)
            errors = [str(m["last_error"]) for m in members if m.get("last_error")]
            detail = (
                f"{sum(int(m.get('processed') or 0) for m in members)} procesados, "
                f"{sum(int(m.get('failures') or 0) for m in members)} fallos"
            )
            if errors:
                detail += f" · último error: {errors[-1]}"
        else:
            state = "unknown"
            detail = "Rol previsto, aún no reportado" if route.planned else "Sin datos"
        nodes.append(
            {
                "id": route.name,
                "label": route.name,
                "state": state,
                "col": col,
                "row": row,
                "instances": len(members),
                "queue_depth": _max_int(m.get("queue_depth") for m in members),
                "restarts": sum(int(m.get("restarts") or 0) for m in members) if members else None,
                "rate_per_s": node_rates.get(route.name),
                "planned": route.planned and not members,
                "detail": detail,
            }
        )

    edges: list[dict[str, Any]] = []
    for route in routes:
        for topic in route.publishes:
            for other in routes:
                if other.name != route.name and topic in other.consumes:
                    topic_stats = stats.get(topic)
                    edges.append(
                        {
                            "id": f"{route.name}>{other.name}:{topic}",
                            "source": route.name,
                            "target": other.name,
                            "topic": topic,
                            "kind": "control" if topic in CONTROL_TOPICS else "data",
                            "rate_per_s": None if topic_stats is None else topic_stats.get("rate_per_s"),
                            "drops": 0 if topic_stats is None else int(topic_stats.get("drops", 0)),
                            "label": _edge_label(topic, topic_stats),
                        }
                    )
    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------
# Feed
# --------------------------------------------------------------------------

_SENSITIVE_KEY = re.compile(
    r"(^|_)(embeddings?|images?|imgs?|jpe?g|png|thumbnails?|crops?|b64|base64|"
    r"biometric|template|pixels?)($|_)|^frame(_data|_bytes)?$",
    re.IGNORECASE,
)


def _looks_like_vector(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 16
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def redact(value: Any) -> Any:
    """Defensa en profundidad: nunca pintar embeddings ni imágenes aunque lleguen."""
    if isinstance(value, Mapping):
        return {
            k: REDACTED if _SENSITIVE_KEY.search(str(k)) else redact(v) for k, v in value.items()
        }
    if _looks_like_vector(value):
        return REDACTED
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def _summary(payload: Mapping[str, Any], limit: int = 140) -> str:
    parts: list[str] = []
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            continue
        parts.append(f"{key}={'—' if value is None else value}")
    text = " ".join(parts) or f"{len(payload)} campos"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def feed_entry(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Una fila del feed: resumen visible y payload expandible (ya redactado)."""
    payload = redact(envelope.get("payload") or {})
    return {
        "event_id": envelope.get("event_id"),
        "topic": envelope.get("topic"),
        "source": envelope.get("source"),
        "stream_id": envelope.get("stream_id"),
        "created_at": envelope.get("created_at"),
        "correlation_id": envelope.get("correlation_id"),
        "causation_id": envelope.get("causation_id"),
        "summary": _summary(payload),
        "payload": payload,
    }


def filter_feed(
    entries: Iterable[Mapping[str, Any]],
    *,
    topics: Iterable[str] | None = None,
    correlation_id: str | None = None,
    text: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    wanted = set(topics) if topics else None
    needle = text.lower() if text else None
    out: list[dict[str, Any]] = []
    for entry in entries:
        if wanted is not None and entry.get("topic") not in wanted:
            continue
        if correlation_id and entry.get("correlation_id") != correlation_id:
            continue
        if needle:
            hay = f"{entry.get('summary', '')} {entry.get('source', '')} {entry.get('topic', '')}"
            if needle not in hay.lower():
                continue
        out.append(dict(entry))
        if limit is not None and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Alertas (decisiones de fusión)
# --------------------------------------------------------------------------

_OUTCOME_LABEL = {
    "alert": "Alerta",
    "inconclusive": "No concluyente",
    "corroborated": "Corroborado",
    "not_restricted": "Zona sin restricción",
    "uncorroborated": "No corroborado",
}
_OUTCOME_SEVERITY = {
    "alert": "high",
    "inconclusive": "medium",
    "uncorroborated": "medium",
    "corroborated": "low",
    "not_restricted": "low",
}
_KIND_LABEL = {
    "track": "Seguimiento",
    "identity": "Identidad",
    "reid": "Re-identificación",
    "location": "Localización",
    "permission": "Permiso",
}
_ROLE_LABEL = {
    "supports": "Respalda",
    "conflicts": "Contradice",
    "stale": "Caducada",
    "context": "Contexto",
}
_REASON_LABEL = {
    "evidence_consistent": "Evidencia consistente",
    "zone_unrestricted": "Zona sin restricción",
    "person_without_tag": "Persona sin tag",
    "tag_without_person": "Tag sin persona visible",
    "hidden_face": "Rostro oculto",
    "stale_sensor": "Sensor sin actualizar",
    "multiple_nearby_people": "Varias personas cerca",
    "unknown_face": "Rostro desconocido",
    "face_inconclusive": "Rostro no concluyente",
    "identity_missing": "Sin identidad",
    "stale_identity": "Identidad caducada",
    "stale_vision": "Visión caducada",
    "identity_tag_mismatch": "Identidad y tag no coinciden",
    "zone_not_permitted": "Zona no permitida",
    "no_permission_record": "Sin registro de permiso",
    "location_invalid": "Localización inválida",
    "clock_skew": "Desfase de reloj",
    "low_confidence": "Confianza baja",
    "zone_unknown": "Zona desconocida",
    "duplicate_track": "Track duplicado",
    "duplicate_identity": "Identidad duplicada",
}


def format_alert(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Decisión de fusión lista para pintar. ``requires_operator`` siempre True."""
    p = redact(envelope.get("payload") or {})
    outcome = str(p.get("outcome", ""))
    confidence = p.get("confidence")
    reasons = [
        {"code": str(code), "label": _REASON_LABEL.get(str(code), str(code))}
        for code in (p.get("reason_codes") or [])
    ]
    evidence = [
        {
            **e,
            "kind_label": _KIND_LABEL.get(str(e.get("kind")), str(e.get("kind", "—"))),
            "role_label": _ROLE_LABEL.get(str(e.get("role")), str(e.get("role", "—"))),
        }
        for e in (p.get("evidence") or [])
        if isinstance(e, Mapping)
    ]
    return {
        "decision_id": p.get("decision_id"),
        "event_id": envelope.get("event_id"),
        "correlation_id": envelope.get("correlation_id"),
        "evaluated_at": p.get("evaluated_at") or envelope.get("created_at"),
        "outcome": outcome,
        "outcome_raw": outcome,
        "outcome_label": _OUTCOME_LABEL.get(
            outcome, "Resultado desconocido" if outcome else "Sin resultado"
        ),
        "severity": _OUTCOME_SEVERITY.get(outcome, "info"),
        "confidence_pct": round(float(confidence) * 100)
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        else None,
        "reasons": reasons,
        "evidence": evidence,
        "stream_id": p.get("stream_id") or envelope.get("stream_id"),
        "zone_id": p.get("zone_id"),
        "person_id": p.get("person_id"),  # tal como llega: seudonimizado
        "track_ref": p.get("track_ref"),
        "requires_operator": True,  # el operador humano siempre es la autoridad final
        "toast": _toast(outcome, p, envelope),
    }


def _toast(outcome: str, p: Mapping[str, Any], envelope: Mapping[str, Any]) -> str:
    where = p.get("zone_id") or p.get("stream_id") or envelope.get("stream_id") or "—"
    head = "Alerta" if outcome == "alert" else _OUTCOME_LABEL.get(outcome, "Decisión")
    return f"{head}: persona en {where}"


def format_alerts(
    envelopes: Iterable[Mapping[str, Any]], limit: int | None = None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for env in envelopes:
        out.append(format_alert(env))
        if limit is not None and len(out) >= limit:
            break
    return out


def is_decision(envelope: Mapping[str, Any]) -> bool:
    payload = envelope.get("payload")
    return isinstance(payload, Mapping) and "decision_id" in payload


# --------------------------------------------------------------------------
# Mapa de sitio y presencia de tags
# --------------------------------------------------------------------------


def parse_site_map(text: str) -> dict[str, Any]:
    """Interpreta ``site_map.toml``: zonas, receptores y cámaras."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"mapa de sitio inválido: {exc}") from exc
    zones = [
        {
            "id": str(z["id"]),
            "name": str(z.get("name", z["id"])),
            "receiver": z.get("receiver"),
            "camera": z.get("camera"),
            "region": list(z["region"]) if z.get("region") else None,
            "calibrated": bool(z.get("calibrated", True)),
        }
        for z in raw.get("zones", [])
        if "id" in z
    ]
    receivers = [
        {"id": str(r["id"]), "kind": r.get("kind"), "zone": r.get("zone")}
        for r in raw.get("receivers", [])
        if "id" in r
    ]
    cameras = [{"id": str(c["id"])} for c in raw.get("cameras", []) if "id" in c]
    return {"zones": zones, "receivers": receivers, "cameras": cameras}


def load_site_map(path: str | Path | None) -> dict[str, Any] | None:
    """Lee el mapa si existe; ``None`` si no hay ruta o archivo (entrada opcional)."""
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        return None
    return parse_site_map(file.read_text(encoding="utf-8"))


def is_location(envelope: Mapping[str, Any]) -> bool:
    """Mensaje de localización de tag, por tópico o por forma del payload."""
    if envelope.get("topic") == "location":
        return True
    payload = envelope.get("payload")
    return isinstance(payload, Mapping) and "tag_ref" in payload and "status" in payload


def tag_presence(envelopes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Último estado por tag. ``envelopes`` va del más nuevo al más antiguo."""
    seen: dict[str, dict[str, Any]] = {}
    for env in envelopes:
        p = env.get("payload") or {}
        tag = p.get("tag_ref")
        if tag is None or str(tag) in seen:
            continue
        confidence = p.get("confidence")
        inside = p.get("status") == "located" and p.get("zone_id") is not None
        seen[str(tag)] = {
            "tag_ref": str(tag),
            "person_ref": p.get("person_ref"),
            "inside": inside,
            "zone_id": p.get("zone_id") if inside else None,
            "confidence_pct": round(float(confidence) * 100)
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            else None,
            "unknown_reason": p.get("unknown_reason"),
            "last_seen": p.get("last_evidence_at") or env.get("created_at"),
        }
    return list(seen.values())


def site_view(
    site: Mapping[str, Any] | None, locations: Iterable[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Zonas con los tags dentro; ``None`` si no hay mapa configurado."""
    if site is None:
        return None
    tags = tag_presence(locations)
    receivers = {r["id"]: r for r in site.get("receivers", [])}
    zones = []
    for zone in site.get("zones", []):
        zones.append(
            {
                **zone,
                "receiver": receivers.get(zone.get("receiver")),
                "tags": [t for t in tags if t["inside"] and t["zone_id"] == zone["id"]],
            }
        )
    return {
        "zones": zones,
        "receivers": list(site.get("receivers", [])),
        "cameras": list(site.get("cameras", [])),
        "outside": [t for t in tags if not t["inside"]],
        "has_location_data": bool(tags),
    }


# --------------------------------------------------------------------------
# Salud
# --------------------------------------------------------------------------


def stream_health(envelopes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Último estado por stream. ``envelopes`` va del más nuevo al más antiguo."""
    seen: dict[str, dict[str, Any]] = {}
    for env in envelopes:
        p = env.get("payload") or {}
        topic = env.get("topic")
        if topic not in ("system.health", "stream.status"):
            continue
        stream = p.get("stream_id") or env.get("stream_id")
        if stream is None or str(stream) in seen:
            continue
        seen[str(stream)] = {
            "stream_id": str(stream),
            "state": str(p.get("state", "sin datos")),
            "fps": p.get("fps"),
            "latency_p50_ms": p.get("latency_ms_p50"),
            "latency_p90_ms": p.get("latency_ms_p90"),
            "frames_received": p.get("frames_received"),
            "frames_dropped": p.get("frames_dropped"),
            "last_error": p.get("last_error"),
        }
    return list(seen.values())


def health_strip(
    api: Mapping[str, Any] | None,
    streams: Sequence[Mapping[str, Any]],
    *,
    connected: bool,
    agents: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Franja de salud: estado de la API, agentes y streams."""
    if api is None or not connected:
        status = "offline"
    else:
        status = str(api.get("status", "ok"))
    api = dict(api or {})
    if agents is not None:  # solo roles: los componentes no cuentan como agentes
        roles = [a for a in agents if a.get("role") != "component"]
        api["agents_total"] = len(roles)
        api["agents_running"] = sum(1 for a in roles if agent_state(a) in ("ok", "degraded"))
        api["agents_failed"] = sum(1 for a in roles if agent_state(a) == "failed")
    return {
        "status": status,
        "api_connected": connected,
        "hardware": HARDWARE_LABEL,
        "agents_total": api.get("agents_total"),
        "agents_running": api.get("agents_running"),
        "agents_failed": api.get("agents_failed"),
        "uptime_s": api.get("uptime_s"),
        "ws_clients": api.get("ws_clients"),
        "ws_dropped_total": api.get("ws_dropped_total"),
        "streams": [dict(s) for s in streams],
    }
