"""Almacén acotado de trazas causales construido sobre el tap del bus.

Indexa los envelopes ya serializados por ``correlation_id`` y reconstruye la
cadena causal (``causation_id``, más las referencias ``evidence_id`` de las
decisiones de fusión) con la latencia de cada salto. Conserva **solo metadata de
trazabilidad** (tópico, origen, ids, marcas de tiempo) y un resumen mínimo de
las decisiones: nunca el payload completo, así que tampoco frames, imágenes ni
embeddings.

Acotado por dos límites: las últimas ``max_traces`` correlaciones (LRU por
último mensaje recibido) y ``max_hops`` saltos por correlación. El índice
``event_id -> correlación`` se depura al expulsar una traza.

La latencia se calcula con ``created_at`` de los envelopes y por tanto supone un
reloj común entre productores; si un salto resulta negativo, la traza se marca
con ``clock_anomaly``.
"""

from __future__ import annotations

import builtins
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any

from bus import Topic

_MAX_ANCESTOR_DEPTH = 64

# Los latidos son periódicos y cada uno abre su propia correlación: llenarían la
# LRU y expulsarían las cadenas que sí importan.
_NOT_CAUSAL = frozenset({Topic.HEALTH})


def _is_decision(topic: Topic, data: dict[str, Any]) -> bool:
    return topic is Topic.EVENTS and "decision_id" in (data.get("payload") or {})


class _Hop:
    __slots__ = (
        "causation_id", "correlation_id", "created_at", "decision", "event_id", "seq",
        "source", "stream_id", "topic", "ts",
    )

    def __init__(self, topic: Topic, data: dict[str, Any], seq: int) -> None:
        self.event_id: str = data["event_id"]
        self.correlation_id: str = data.get("correlation_id") or self.event_id
        self.causation_id: str | None = data.get("causation_id")
        self.topic = topic
        self.source: str = data["source"]
        self.stream_id: str | None = data.get("stream_id")
        self.created_at: str = data["created_at"]
        self.ts = datetime.fromisoformat(self.created_at)
        self.seq = seq
        self.decision: dict[str, Any] | None = None
        if _is_decision(topic, data):
            p = data["payload"]
            self.decision = {
                "decision_id": p.get("decision_id"),
                "outcome": p.get("outcome"),
                "confidence": p.get("confidence"),
                "reason_codes": list(p.get("reason_codes") or []),
                "evidence": [
                    {
                        "evidence_id": e.get("evidence_id"),
                        "kind": e.get("kind"),
                        "role": e.get("role"),
                    }
                    for e in (p.get("evidence") or [])
                    if isinstance(e, dict) and e.get("evidence_id")
                ],
            }


class _Trace:
    __slots__ = ("hops", "truncated")

    def __init__(self) -> None:
        self.hops: dict[str, _Hop] = {}
        self.truncated = False


class TraceStore:
    """Trazas por correlación, acotadas y de solo metadata."""

    def __init__(self, max_traces: int = 200, max_hops: int = 500) -> None:
        if max_traces < 1:
            raise ValueError("max_traces debe ser al menos 1")
        if max_hops < 1:
            raise ValueError("max_hops debe ser al menos 1")
        self._max_traces = max_traces
        self._max_hops = max_hops
        self._traces: OrderedDict[str, _Trace] = OrderedDict()
        self._index: dict[str, str] = {}  # event_id -> correlation_id
        self._seq = 0
        self._lock = threading.Lock()

    # -- escritura (firma compatible con ``BusTap.add_listener``) --

    def record(self, topic: Topic, data: dict[str, Any]) -> None:
        topic = Topic(topic)
        if topic in _NOT_CAUSAL:
            return
        with self._lock:
            hop = _Hop(topic, data, self._seq)
            self._seq += 1
            trace = self._traces.get(hop.correlation_id)
            if trace is None:
                trace = _Trace()
                self._traces[hop.correlation_id] = trace
            self._traces.move_to_end(hop.correlation_id)
            if hop.event_id not in trace.hops:
                if len(trace.hops) >= self._max_hops:
                    trace.truncated = True
                else:
                    trace.hops[hop.event_id] = hop
                    self._index[hop.event_id] = hop.correlation_id
            while len(self._traces) > self._max_traces:
                old_id, old = self._traces.popitem(last=False)
                for event_id in old.hops:
                    if self._index.get(event_id) == old_id:
                        del self._index[event_id]

    # -- lectura --

    def get(self, correlation_id: str) -> dict[str, Any] | None:
        with self._lock:
            trace = self._traces.get(correlation_id)
            if trace is None:
                return None
            return self._build(correlation_id, trace)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Resúmenes (sin saltos), la correlación más reciente primero."""
        with self._lock:
            out: list[dict[str, Any]] = []
            for correlation_id in reversed(self._traces):
                built = self._build(correlation_id, self._traces[correlation_id])
                out.append(
                    {
                        "correlation_id": correlation_id,
                        "started_at": built["started_at"],
                        "ended_at": built["ended_at"],
                        "hops": len(built["hops"]),
                        "topics": built["topics"],
                        "has_alert": built["has_alert"],
                        "end_to_end_ms": built["end_to_end_ms"],
                        "decision_ids": [d["decision_id"] for d in built["decisions"]],
                        "truncated": built["truncated"],
                        "clock_anomaly": built["clock_anomaly"],
                    }
                )
                if len(out) >= limit:
                    break
            return out

    # -- construcción de la cadena --

    def _hop(self, event_id: str) -> _Hop | None:
        correlation_id = self._index.get(event_id)
        if correlation_id is None:
            return None
        trace = self._traces.get(correlation_id)
        return None if trace is None else trace.hops.get(event_id)

    def _ancestors(self, start: _Hop, acc: dict[str, _Hop], depth: int = 0) -> None:
        """Añade a ``acc`` el salto y todo lo que lo causó (causación y evidencia)."""
        if start.event_id in acc or depth > _MAX_ANCESTOR_DEPTH:
            return
        acc[start.event_id] = start
        if start.causation_id:
            parent = self._hop(start.causation_id)
            if parent is not None:
                self._ancestors(parent, acc, depth + 1)
        if start.decision is not None:
            for evidence in start.decision["evidence"]:
                parent = self._hop(evidence["evidence_id"])
                if parent is not None:
                    self._ancestors(parent, acc, depth + 1)

    @staticmethod
    def _parents(hop: _Hop, combined: dict[str, _Hop]) -> builtins.list[str]:
        parents: builtins.list[str] = []
        if hop.causation_id and hop.causation_id in combined:
            parents.append(hop.causation_id)
        if hop.decision is not None:
            for evidence in hop.decision["evidence"]:
                eid = evidence["evidence_id"]
                if eid in combined and eid not in parents:
                    parents.append(eid)
        return parents

    @staticmethod
    def _causal_order(
        combined: dict[str, _Hop], parents_of: dict[str, builtins.list[str]]
    ) -> builtins.list[_Hop]:
        """Orden por tiempo, pero sin poner nunca a un hijo antes que su padre.

        Con marcas iguales (reloj grueso) o llegadas desordenadas, el simple
        orden por tiempo podría invertir una arista causal.
        """
        pending = sorted(combined.values(), key=lambda h: (h.ts, h.seq))
        placed: set[str] = set()
        out: builtins.list[_Hop] = []
        while pending:
            pick = next(
                (h for h in pending if all(p in placed for p in parents_of[h.event_id])),
                pending[0],  # ciclo imposible en la práctica: no bloquear
            )
            pending.remove(pick)
            placed.add(pick.event_id)
            out.append(pick)
        return out

    def _build(self, correlation_id: str, trace: _Trace) -> dict[str, Any]:
        combined: dict[str, _Hop] = {}
        for hop in trace.hops.values():
            self._ancestors(hop, combined)
        parents_of = {h.event_id: self._parents(h, combined) for h in combined.values()}
        ordered = self._causal_order(combined, parents_of)
        start = ordered[0].ts if ordered else None

        hops: builtins.list[dict[str, Any]] = []
        anomaly = False
        for hop in ordered:
            parents = parents_of[hop.event_id]
            latency: float | None = None
            if parents:
                latest = max(combined[p].ts for p in parents)
                latency = round((hop.ts - latest).total_seconds() * 1000, 3)
                anomaly = anomaly or latency < 0
            assert start is not None
            hops.append(
                {
                    "event_id": hop.event_id,
                    "correlation_id": hop.correlation_id,
                    "topic": hop.topic.value,
                    "source": hop.source,
                    "stream_id": hop.stream_id,
                    "causation_id": hop.causation_id,
                    "parent_event_ids": parents,
                    "created_at": hop.created_at,
                    "hop_latency_ms": latency,
                    "since_start_ms": round((hop.ts - start).total_seconds() * 1000, 3),
                    "via": "own" if hop.correlation_id == correlation_id else "upstream",
                }
            )

        alerts = [h for h in ordered if h.topic is Topic.EVENTS]
        end_to_end: float | None = None
        if alerts and start is not None:
            end_to_end = round((alerts[-1].ts - start).total_seconds() * 1000, 3)

        decisions = []
        for hop in ordered:
            if hop.decision is None or hop.correlation_id != correlation_id:
                continue
            decisions.append(
                {
                    **hop.decision,
                    "event_id": hop.event_id,
                    "evidence": [
                        {**e, "resolved": e["evidence_id"] in combined}
                        for e in hop.decision["evidence"]
                    ],
                }
            )

        return {
            "correlation_id": correlation_id,
            "started_at": ordered[0].created_at if ordered else None,
            "ended_at": ordered[-1].created_at if ordered else None,
            "has_alert": bool(alerts),
            "end_to_end_ms": end_to_end,
            "clock_anomaly": anomaly,
            "truncated": trace.truncated,
            "topics": sorted({h.topic.value for h in ordered}),
            "hops": hops,
            "decisions": decisions,
        }
