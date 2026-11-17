"""Integración con Langfuse: opcional, solo por entorno, solo metadata filtrada."""

import json
from datetime import UTC, datetime

import pytest

from bus import MetadataEnvelope, Topic, envelope_to_dict
from tracing import PrivacyFilter, TraceStore
from tracing.langfuse_sink import (
    LangfuseConfig,
    LangfuseDecisionTracer,
    create_tracer_from_env,
)
from tracing.privacy import PrivacyConfigError


class FakeSpan:
    def __init__(self, calls, kwargs):
        self.calls = calls
        self.kwargs = kwargs
        calls.append(("start", kwargs))

    def start_observation(self, **kwargs):
        return FakeSpan(self.calls, kwargs)

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return self

    def end(self):
        self.calls.append(("end", {}))
        return self


class FakeClient:
    def __init__(self):
        self.calls = []
        self.flushed = 0
        self.shutdowns = 0

    def create_trace_id(self, *, seed=None):
        return f"trace-{seed}"

    def start_observation(self, **kwargs):
        return FakeSpan(self.calls, kwargs)

    def flush(self):
        self.flushed += 1

    def shutdown(self):
        self.shutdowns += 1


def record(store, tracer, topic, envelope):
    data = envelope_to_dict(topic, envelope)
    store.record(topic, data)
    tracer.on_message(topic, data)


def decision_envelope(**extra):
    payload = {
        "decision_id": "dec-1", "outcome": "alert", "confidence": 0.5,
        "reason_codes": ["unknown_face"], "zone_id": "z1", "requires_operator": True,
        "person_id": "p-7f3a",
        "evidence": [{"evidence_id": "x", "kind": "identity", "role": "supports"}],
    }
    payload.update(extra)
    return MetadataEnvelope(
        source="fusion", stream_id="cam-1", payload=payload, event_id="dec-1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


# -- configuración --------------------------------------------------------


def test_sin_claves_queda_desactivado():
    assert LangfuseConfig.from_env({}) is None
    assert LangfuseConfig.from_env({"LANGFUSE_PUBLIC_KEY": "pk"}) is None
    assert LangfuseConfig.from_env({"LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_HOST": "h"}) is None


def test_sin_host_explicito_no_se_activa_para_no_enviar_a_la_nube_por_defecto():
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
    assert LangfuseConfig.from_env(env) is None


def test_config_desde_entorno_acepta_host_o_base_url():
    base = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
    a = LangfuseConfig.from_env({**base, "LANGFUSE_HOST": "http://localhost:3000"})
    b = LangfuseConfig.from_env({**base, "LANGFUSE_BASE_URL": "http://localhost:3001"})
    assert a is not None and a.host == "http://localhost:3000"
    assert b is not None and b.host == "http://localhost:3001"


def test_repr_de_config_no_filtra_secretos():
    cfg = LangfuseConfig("pk-lf-123", "sk-lf-999", "http://localhost:3000")
    assert "sk-lf-999" not in repr(cfg) and "pk-lf-123" not in repr(cfg)


def test_clave_hmac_invalida_falla_con_error_claro_antes_de_crear_el_cliente(monkeypatch):
    import tracing.langfuse_sink as sink

    monkeypatch.setattr(sink, "_make_client", lambda c: pytest.fail("no debe crear cliente"))
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk",
           "LANGFUSE_HOST": "http://h", "TRACING_HASH_KEY": "corta"}
    with pytest.raises(PrivacyConfigError, match="TRACING_HASH_KEY"):
        create_tracer_from_env(TraceStore(), env=env)


def test_end_to_end_no_finito_no_se_exporta():
    class Odd(TraceStore):
        def get(self, correlation_id):
            return {"hops": [], "end_to_end_ms": float("inf")}

    client = FakeClient()
    tracer = LangfuseDecisionTracer(client, Odd())
    tracer.on_message(Topic.EVENTS, envelope_to_dict(Topic.EVENTS, decision_envelope()))
    out = next(kw for kind, kw in client.calls if kind == "start")["output"]
    assert out["end_to_end_ms"] is None


def test_create_tracer_sin_claves_devuelve_none():
    assert create_tracer_from_env(TraceStore(), env={}) is None


def test_create_tracer_sin_sdk_instalado_se_degrada(monkeypatch):
    import tracing.langfuse_sink as sink

    def boom(config):
        raise ImportError("langfuse")

    monkeypatch.setattr(sink, "_make_client", boom)
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_HOST": "http://h"}
    assert create_tracer_from_env(TraceStore(), env=env) is None


# -- qué se envía ---------------------------------------------------------


def test_una_traza_por_decision_de_fusion_con_hijos_de_evidencia():
    client, store = FakeClient(), TraceStore()
    privacy = PrivacyFilter(b"k" * 32)
    tracer = LangfuseDecisionTracer(client, store, privacy)
    record(store, tracer, Topic.EVENTS, decision_envelope())
    starts = [kw for kind, kw in client.calls if kind == "start"]
    assert starts[0]["name"] == "fusion.decision"
    # el seed es el id HMAC: el id crudo nunca sale (ni como semilla)
    assert starts[0]["trace_context"] == {"trace_id": f"trace-{privacy.hash('dec-1')}"}
    assert "dec-1" not in json.dumps(client.calls, default=str)
    assert starts[0]["output"]["outcome"] == "alert"
    assert {s["name"] for s in starts[1:]} >= {"evidence.identity"}
    assert ("end", {}) in client.calls


def test_evento_de_regla_y_comando_del_supervisor_se_trazan_como_reglas():
    client, store = FakeClient(), TraceStore()
    tracer = LangfuseDecisionTracer(client, store)
    evt = MetadataEnvelope(source="event", payload={"type": "zone_intrusion", "zone_id": "z1"})
    cmd = MetadataEnvelope(source="supervisor", payload={"action": "restart", "target": "tracker"})
    other = MetadataEnvelope(source="tracker", payload={"tracks": []})
    record(store, tracer, Topic.EVENTS, evt)
    record(store, tracer, Topic.COMMANDS, cmd)
    record(store, tracer, Topic.TRACKS, other)
    names = [kw["name"] for kind, kw in client.calls if kind == "start" and "trace_context" in kw]
    assert names == ["event.rule", "supervisor.command"]
    for kind, kw in client.calls:
        if kind == "start" and kw["name"] in ("event.rule", "supervisor.command"):
            assert kw["metadata"]["decision_engine"] == "rules"
            assert kw["metadata"]["llm"] is False


def test_el_contexto_causal_viaja_en_la_traza():
    client, store = FakeClient(), TraceStore()
    tracer = LangfuseDecisionTracer(client, store)
    det = MetadataEnvelope(source="inference", payload={"detections": []}, event_id="d")
    trk = det.derive("tracker", {"tracks": []}, suffix="tracks")
    evt = trk.derive("event", {"type": "zone_intrusion", "zone_id": "z"}, suffix="events/0")
    record(store, tracer, Topic.DETECTIONS, det)
    record(store, tracer, Topic.TRACKS, trk)
    record(store, tracer, Topic.EVENTS, evt)
    root = next(kw for kind, kw in client.calls if kind == "start" and kw["name"] == "event.rule")
    chain = root["input"]["chain"]
    assert [h["topic"] for h in chain] == ["vision.detections", "vision.tracks", "events"]
    assert root["output"]["end_to_end_ms"] is not None


def test_nada_biometrico_ni_en_claro_llega_al_cliente():
    client, store = FakeClient(), TraceStore()
    tracer = LangfuseDecisionTracer(client, store)
    hostil = decision_envelope(
        embedding=[0.123456] * 512,
        image="data:image/jpeg;base64," + "A" * 400,
        tag_id="AA:BB:CC:DD:EE:FF",
        person_id="Juan Perez",
        name="Juan Perez",
        evidence=[
            {"evidence_id": "x", "kind": "identity", "role": "supports",
             "embedding": [0.987654] * 8, "detail": "Juan Perez tag AA:BB:CC:DD:EE:FF"}
        ],
    )
    record(store, tracer, Topic.EVENTS, hostil)
    record(
        store, tracer, Topic.EVENTS,
        MetadataEnvelope(source="event", payload={"type": "x", "tag_id": "04A224B2", "face": "f.jpg"}),
    )
    text = json.dumps(client.calls, default=str)
    for secreto in (
        "embedding", "0.123456", "0.987654", "data:image", "AA:BB:CC", "Juan Perez",
        "04A224B2", "f.jpg", "tag_id",
    ):
        assert secreto not in text, secreto
    assert "p-7f3a" not in text and "h-" in text  # solo HMAC


def test_un_fallo_del_cliente_no_rompe_el_bus():
    class Broken(FakeClient):
        def start_observation(self, **kwargs):
            raise RuntimeError("langfuse caído")

    store = TraceStore()
    tracer = LangfuseDecisionTracer(Broken(), store)
    record(store, tracer, Topic.EVENTS, decision_envelope())  # no lanza


def test_ids_no_confiables_del_envelope_y_la_cadena_se_hashean():
    client, store = FakeClient(), TraceStore()
    tracer = LangfuseDecisionTracer(client, store)
    evt = MetadataEnvelope(
        source="event", stream_id="employee-4411@corp",
        payload={"type": "x"}, event_id="employee-4411@corp", correlation_id="jperez@corp",
    )
    record(store, tracer, Topic.EVENTS, evt)
    text = json.dumps(client.calls, default=str)
    assert "employee" not in text and "jperez" not in text and "corp" not in text


def test_reentrega_no_duplica_la_traza():
    client, store = FakeClient(), TraceStore()
    tracer = LangfuseDecisionTracer(client, store)
    env = decision_envelope()
    record(store, tracer, Topic.EVENTS, env)
    n = len(client.calls)
    record(store, tracer, Topic.EVENTS, env)  # misma event_id
    assert len(client.calls) == n


def test_memoria_de_deduplicacion_acotada():
    client, store = FakeClient(), TraceStore()
    tracer = LangfuseDecisionTracer(client, store, dedupe_size=2)
    for i in range(3):
        record(store, tracer, Topic.EVENTS,
               MetadataEnvelope(source="event", payload={"type": "x"}, event_id=f"{i:08x}"))
    assert len(tracer._seen) == 2


def test_close_hace_flush_y_shutdown():
    client = FakeClient()
    tracer = LangfuseDecisionTracer(client, TraceStore())
    tracer.close()
    assert client.flushed == 1 and client.shutdowns == 1


# -- SDK real (sin red: exportador en memoria) ------------------------------


def test_sdk_real_con_exportador_en_memoria():
    pytest.importorskip("langfuse")
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-test", secret_key="sk-test", host="http://localhost:9",
        span_exporter=exporter, tracing_enabled=True,
    )
    store = TraceStore()
    tracer = LangfuseDecisionTracer(client, store)
    record(store, tracer, Topic.EVENTS, decision_envelope(embedding=[0.5] * 16))
    tracer.close()
    names = {s.name for s in exporter.get_finished_spans()}
    assert "fusion.decision" in names
    attrs = json.dumps([dict(s.attributes or {}) for s in exporter.get_finished_spans()], default=str)
    assert "embedding" not in attrs
