"""Pruebas del almacén de trazas causales (metadata, acotado, latencias por salto)."""

from datetime import UTC, datetime, timedelta

import pytest

from bus import MetadataEnvelope, Topic, envelope_to_dict
from tracing import TraceStore

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def ms(n: int) -> datetime:
    return T0 + timedelta(milliseconds=n)


def feed(store, topic, envelope):
    store.record(topic, envelope_to_dict(topic, envelope))
    return envelope


def root(at_ms, source="ingest", payload=None, event_id="d1", stream="cam-1"):
    return MetadataEnvelope(
        source=source, payload=payload or {"detections": []}, stream_id=stream,
        created_at=ms(at_ms), event_id=event_id,
    )


def child(parent, at_ms, source, suffix, payload=None):
    env = parent.derive(source, payload or {"k": 1}, suffix=suffix)
    return MetadataEnvelope(
        source=env.source, payload=env.payload, stream_id=env.stream_id,
        created_at=ms(at_ms), event_id=env.event_id,
        correlation_id=env.correlation_id, causation_id=env.causation_id,
    )


def build_chain(store):
    det = feed(store, Topic.DETECTIONS, root(0))
    trk = feed(store, Topic.TRACKS, child(det, 40, "tracker", "tracks"))
    evt = feed(
        store, Topic.EVENTS,
        child(trk, 100, "event", "events/0", {"type": "person_in_zone", "zone_id": "z1"}),
    )
    return det, trk, evt


def test_cadena_ordenada_con_latencia_por_salto_y_total():
    store = TraceStore()
    det, trk, _ = build_chain(store)
    trace = store.get(det.correlation_id)
    assert trace is not None
    assert [h["topic"] for h in trace["hops"]] == ["vision.detections", "vision.tracks", "events"]
    assert [h["hop_latency_ms"] for h in trace["hops"]] == [None, 40.0, 60.0]
    assert [h["causation_id"] for h in trace["hops"]] == [None, det.event_id, trk.event_id]
    assert trace["hops"][2]["source"] == "event"
    assert trace["has_alert"] is True
    assert trace["end_to_end_ms"] == 100.0
    assert trace["clock_anomaly"] is False


def test_sin_evento_final_no_hay_latencia_de_alerta():
    store = TraceStore()
    det = feed(store, Topic.DETECTIONS, root(0))
    feed(store, Topic.TRACKS, child(det, 10, "tracker", "tracks"))
    trace = store.get(det.correlation_id)
    assert trace["has_alert"] is False
    assert trace["end_to_end_ms"] is None


def test_orden_por_tiempo_aunque_llegue_desordenado():
    store = TraceStore()
    det = root(0)
    trk = child(det, 40, "tracker", "tracks")
    feed(store, Topic.TRACKS, trk)  # llega antes que su causa
    feed(store, Topic.DETECTIONS, det)
    trace = store.get(det.correlation_id)
    assert [h["event_id"] for h in trace["hops"]] == [det.event_id, trk.event_id]
    assert trace["hops"][1]["hop_latency_ms"] == 40.0


def test_padre_desconocido_deja_latencia_nula():
    store = TraceStore()
    det = root(0)
    trk = child(det, 40, "tracker", "tracks")
    feed(store, Topic.TRACKS, trk)  # el padre nunca llegó
    hop = store.get(trk.correlation_id)["hops"][0]
    assert hop["hop_latency_ms"] is None
    assert hop["causation_id"] == det.event_id


def test_reloj_inconsistente_se_marca():
    store = TraceStore()
    det = feed(store, Topic.DETECTIONS, root(100))
    feed(store, Topic.TRACKS, child(det, 50, "tracker", "tracks"))
    assert store.get(det.correlation_id)["clock_anomaly"] is True


def test_lru_acota_las_correlaciones():
    store = TraceStore(max_traces=2)
    for i in range(3):
        feed(store, Topic.DETECTIONS, root(i, event_id=f"c{i}"))
    ids = [t["correlation_id"] for t in store.list(10)]
    assert ids == ["c2", "c1"]  # más nuevo primero; c0 expulsada
    assert store.get("c0") is None


def test_lru_conserva_la_correlacion_recien_tocada():
    store = TraceStore(max_traces=2)
    a = feed(store, Topic.DETECTIONS, root(0, event_id="a"))
    b = feed(store, Topic.DETECTIONS, root(1, event_id="b"))
    feed(store, Topic.TRACKS, child(a, 2, "tracker", "tracks"))  # toca 'a'
    feed(store, Topic.DETECTIONS, root(3, event_id="c"))
    assert store.get("a") is not None
    assert store.get(b.correlation_id) is None


def test_saltos_por_traza_acotados():
    store = TraceStore(max_hops=3)
    det = feed(store, Topic.DETECTIONS, root(0))
    for i in range(5):
        feed(store, Topic.TRACKS, child(det, 10 + i, "tracker", f"t{i}"))
    trace = store.get(det.correlation_id)
    assert len(trace["hops"]) == 3
    assert trace["truncated"] is True


def test_list_respeta_limite_y_resume_sin_payload():
    store = TraceStore()
    build_chain(store)
    feed(store, Topic.DETECTIONS, root(500, event_id="z9"))
    rows = store.list(1)
    assert len(rows) == 1 and rows[0]["correlation_id"] == "z9"
    chain = next(r for r in store.list(10) if r["correlation_id"] == "d1")
    assert chain["hops"] == 3 and chain["has_alert"] is True and chain["end_to_end_ms"] == 100.0
    assert "payload" not in chain


def test_los_saltos_no_exponen_payload():
    store = TraceStore()
    det, _, _ = build_chain(store)
    for hop in store.get(det.correlation_id)["hops"]:
        assert "payload" not in hop


def test_decision_enlaza_evidencia_y_hereda_la_cadena_ascendente():
    store = TraceStore()
    det, trk, _ = build_chain(store)
    decision = MetadataEnvelope(
        source="fusion", stream_id="cam-1", created_at=ms(180), event_id="dec-1",
        payload={
            "decision_id": "dec-1", "outcome": "uncorroborated", "confidence": 0.5,
            "reason_codes": ["unknown_face"], "zone_id": "z1", "requires_operator": True,
            "evidence": [
                {"evidence_id": trk.event_id, "kind": "track", "role": "supports"},
                {"evidence_id": "no-existe", "kind": "identity", "role": "context"},
            ],
        },
    )
    feed(store, Topic.EVENTS, decision)
    trace = store.get("dec-1")
    assert [h["event_id"] for h in trace["hops"]] == [det.event_id, trk.event_id, "dec-1"]
    last = trace["hops"][-1]
    assert last["parent_event_ids"] == [trk.event_id]
    assert last["hop_latency_ms"] == 140.0  # desde la evidencia más reciente (40 ms)
    assert trace["has_alert"] is True
    assert trace["end_to_end_ms"] == 180.0  # desde la raíz de la cadena ascendente
    (dec,) = trace["decisions"]
    assert dec["decision_id"] == "dec-1" and dec["outcome"] == "uncorroborated"
    assert [(e["evidence_id"], e["resolved"]) for e in dec["evidence"]] == [
        (trk.event_id, True),
        ("no-existe", False),
    ]


def test_los_latidos_de_salud_no_ocupan_trazas():
    store = TraceStore(max_traces=2)
    feed(store, Topic.DETECTIONS, root(0, event_id="a"))
    for i in range(10):
        feed(store, Topic.HEALTH, root(i, event_id=f"hb{i}", source="tracker"))
    assert store.get("a") is not None
    assert store.get("hb0") is None
    assert [t["correlation_id"] for t in store.list(10)] == ["a"]


def test_con_marcas_iguales_el_padre_va_antes_que_el_hijo():
    store = TraceStore()
    det = root(0)
    decision = MetadataEnvelope(
        source="fusion", created_at=ms(0), event_id="dec-0", stream_id="cam-1",
        payload={
            "decision_id": "dec-0",
            "evidence": [
                {"evidence_id": f"{det.event_id}/tracks", "kind": "track", "role": "supports"}
            ],
        },
    )
    feed(store, Topic.EVENTS, decision)  # llega antes que su evidencia (misma marca)
    feed(store, Topic.DETECTIONS, det)
    feed(store, Topic.TRACKS, child(det, 0, "tracker", "tracks"))
    ids = [h["event_id"] for h in store.get("dec-0")["hops"]]
    assert ids == [det.event_id, f"{det.event_id}/tracks", "dec-0"]


def raw(event_id, created_at, correlation_id=None, causation_id=None, topic="events"):
    return {
        "topic": topic, "source": "s", "payload": {}, "stream_id": None,
        "created_at": created_at, "event_id": event_id,
        "correlation_id": correlation_id or event_id, "causation_id": causation_id,
    }


def test_marcas_ingenuas_y_con_zona_no_rompen_get_ni_list():
    store = TraceStore()
    store.record(Topic.DETECTIONS, raw("a", "2026-01-01T00:00:00+00:00", "c", topic="vision.detections"))
    store.record(Topic.EVENTS, raw("b", "2026-01-01T00:00:00.050", "c", "a"))  # sin zona horaria
    trace = store.get("c")
    assert [h["event_id"] for h in trace["hops"]] == ["a", "b"]
    assert trace["hops"][1]["hop_latency_ms"] == 50.0  # la ingenua se toma como UTC
    assert trace["hops"][1]["created_at"].endswith("+00:00")
    assert store.list(10)[0]["hops"] == 2


def test_zonas_horarias_distintas_se_normalizan_a_utc():
    store = TraceStore()
    store.record(Topic.DETECTIONS, raw("a", "2026-01-01T01:00:00+01:00", "c", topic="vision.detections"))
    store.record(Topic.EVENTS, raw("b", "2026-01-01T00:00:00.020+00:00", "c", "a"))
    assert store.get("c")["hops"][1]["hop_latency_ms"] == 20.0


@pytest.mark.parametrize("malo", ["ayer", "", None, 12, "2026-13-45T00:00:00"])
def test_marca_invalida_se_descarta_y_se_cuenta_sin_lanzar(malo):
    store = TraceStore()
    store.record(Topic.EVENTS, raw("a", malo, "c"))
    store.record(Topic.EVENTS, raw("b", "2026-01-01T00:00:00+00:00", "c2"))
    assert store.skipped == 1
    assert store.get("c") is None
    assert [t["correlation_id"] for t in store.list(10)] == ["c2"]


def test_mensaje_sin_campos_se_descarta_y_se_cuenta():
    store = TraceStore()
    store.record(Topic.EVENTS, {"topic": "events"})
    assert store.skipped == 1


def test_list_de_50_trazas_de_500_saltos_es_rapido():
    import time

    store = TraceStore(max_traces=60, max_hops=500)
    for c in range(50):
        parent = None
        for i in range(500):
            eid = f"{c}-{i}"
            store.record(
                Topic.TRACKS,
                raw(eid, f"2026-01-01T00:00:{i // 100:02d}.{(i % 100) * 10:03d}+00:00",
                    f"c{c}", parent, topic="vision.tracks"),
            )
            parent = eid
    t0 = time.perf_counter()
    rows = store.list(50)
    elapsed = time.perf_counter() - t0
    assert len(rows) == 50 and rows[0]["hops"] == 500
    assert elapsed < 0.5, elapsed
    t0 = time.perf_counter()
    store.get("c3")
    assert time.perf_counter() - t0 < 0.1
