"""Filtro de privacidad previo a Langfuse: deny-by-default, sin biometría ni IDs en claro."""

import json
import math

import pytest

from tracing.privacy import AGENT_ROLES, PrivacyConfigError, PrivacyFilter

KEY = b"k" * 32

HOSTIL = {
    "decision_id": "dec-0001",
    "outcome": "alert",
    "confidence": 0.62,
    "reason_codes": ["unknown_face"],
    "zone_id": "bodega",
    "stream_id": "cam-1",
    "person_id": "p-7f3a",
    "requires_operator": True,
    "embedding": [0.01] * 512,
    "face_embedding": [0.2, 0.3],
    "image": "data:image/jpeg;base64," + "A" * 400,
    "frame": "QUJD" * 100,
    "face_crop": "crop.jpg",
    "tag_id": "AA:BB:CC:DD:EE:FF",
    "rfid": "04A224B2",
    "name": "Juan Perez",
    "detections": [{"bbox": [1, 2, 3, 4], "embedding": [0.1] * 8}],
    "evidence": [
        {
            "evidence_id": "e1", "kind": "identity", "role": "supports",
            "confidence": 0.9, "embedding": [0.5] * 64, "detail": "Juan Perez, tag 04A224B2",
            "zone_id": "bodega",
        }
    ],
}

# Ids que antes pasaban por "tener forma de sistema" y llevan datos personales.
IDS_HOSTILES = [
    "6a6f686e2e736d69746840636f72702e636f6d",  # email en hex
    "restart/JohnSmith/Madrid",
    "dec-1/JohnSmith/Ana-Lopez",
    "deadbeef/Juan-Perez-DNI12345678",
    "employee-4411@corp",
    "0a2093bbd06f4a3db2be4e4e49dd6727/tracks",  # incluso uno legítimo se hashea
    "dec-0001",
]


def dump(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def flt(**kw) -> PrivacyFilter:
    return PrivacyFilter(KEY, **kw)


def is_hash(value) -> bool:
    return isinstance(value, str) and value.startswith("h-") and len(value) == 18


# -- identificadores: SIEMPRE HMAC --------------------------------------------


@pytest.mark.parametrize("valor", IDS_HOSTILES)
def test_todo_identificador_sale_como_hmac(valor):
    f = flt()
    out = f.payload(
        {"decision_id": valor, "track_ref": valor, "person_id": valor,
         "evidence": [{"evidence_id": valor, "kind": "track", "role": "context"}]}
    )
    assert is_hash(out["decision_id"]) and is_hash(out["track_ref"]) and is_hash(out["person_id"])
    assert is_hash(out["evidence"][0]["evidence_id"])
    assert valor not in dump(out)
    env = f.envelope(
        {"topic": "events", "source": "event", "payload": {}, "event_id": valor,
         "correlation_id": valor, "causation_id": valor, "created_at": "2026-01-01T00:00:00+00:00"}
    )
    assert all(is_hash(env[k]) for k in ("event_id", "correlation_id", "causation_id"))
    hop = f.hop({"event_id": valor, "causation_id": valor, "parent_event_ids": [valor],
                 "topic": "events", "source": "event"})
    assert is_hash(hop["event_id"]) and is_hash(hop["causation_id"])
    assert all(is_hash(p) for p in hop["parent_event_ids"])
    assert valor not in dump([env, hop])


def test_el_hash_enlaza_igual_en_payload_envelope_y_hop():
    f = flt()
    a = f.payload({"decision_id": "x1"})["decision_id"]
    b = f.envelope({"event_id": "x1", "payload": {}})["event_id"]
    c = f.hop({"event_id": "x1"})["event_id"]
    assert a == b == c


def test_el_hash_es_estable_con_la_misma_clave_y_distinto_con_otra():
    a = PrivacyFilter(KEY).payload({"person_id": "p-1234"})["person_id"]
    b = PrivacyFilter(KEY).payload({"person_id": "p-1234"})["person_id"]
    c = PrivacyFilter(b"z" * 32).payload({"person_id": "p-1234"})["person_id"]
    assert a == b and a != c


def test_sin_clave_se_genera_una_aleatoria_por_proceso():
    a = PrivacyFilter().payload({"person_id": "p-1"})["person_id"]
    b = PrivacyFilter().payload({"person_id": "p-1"})["person_id"]
    assert a != b


# -- clave -------------------------------------------------------------------


def test_clave_demasiado_corta_se_rechaza_con_error_claro():
    with pytest.raises(PrivacyConfigError, match="al menos 16"):
        PrivacyFilter(b"corta")
    with pytest.raises(PrivacyConfigError, match="TRACING_HASH_KEY"):
        PrivacyFilter.from_env({"TRACING_HASH_KEY": "corta"})


def test_clave_valida_desde_entorno():
    f = PrivacyFilter.from_env({"TRACING_HASH_KEY": "x" * 16})
    assert is_hash(f.payload({"person_id": "p"})["person_id"])


# -- etiquetas: vocabularios cerrados por campo ----------------------------------


@pytest.mark.parametrize(
    ("campo", "valido"),
    [
        ("type", "loitering"), ("type", "zone_intrusion"), ("action", "restart"),
        ("target", "tracker"), ("state", "running"), ("state", "failed"),
        ("state", "up"), ("stage", "queue_overflow"), ("stage", "handle"),
        ("status", "match"), ("role", "supervisor"), ("error_type", "ValueError"),
    ],
)
def test_vocabulario_acepta_valores_que_el_proyecto_emite(campo, valido):
    assert flt().payload({campo: valido})[campo] == valido


@pytest.mark.parametrize("campo", ["type", "action", "target", "state", "stage", "status", "role", "error_type"])
@pytest.mark.parametrize(
    "malo",
    ["maria_gonzalez_dni_123456", "employee-4411@corp", "Data:xxx", "a" * 70, "x", "loitering\n", 7, ["a"], None],
)
def test_etiquetas_libres_se_descartan_y_se_cuentan(campo, malo):
    f = flt()
    out = f.payload({campo: malo})
    assert campo not in out and out["redacted_fields"] == 1
    assert "maria_gonzalez" not in dump(out)


def test_vocabulario_de_roles_coincide_con_las_rutas_reales():
    from agents.route import MULTIAGENT_ROUTE

    assert set(AGENT_ROLES) == {r.name for r in MULTIAGENT_ROUTE}


def test_outcome_y_kind_son_vocabulario_cerrado():
    out = flt().payload({"outcome": "juan_perez_4411", "evidence": [
        {"evidence_id": "e1", "kind": "juan", "role": "supports"}]})
    assert "outcome" not in out
    assert out["evidence"][0].get("kind") is None


def test_reason_codes_solo_vocabulario_cerrado_no_floats():
    out = flt().payload({"reason_codes": [0.1] * 512})
    assert "reason_codes" not in out
    out = flt().payload({"reason_codes": ["unknown_face", "made_up_code", 3, "A" * 80, "x@y", ["a"]]})
    assert out["reason_codes"] == ["unknown_face"]


# -- stream / zone / source: HMAC salvo allowlist explícita -------------------------


def test_stream_zone_y_source_se_hashean_si_no_estan_en_la_allowlist():
    f = flt()
    out = f.payload({"stream_id": "employee-4411", "zone_id": "Juan.Perez",
                     "evidence": [{"evidence_id": "e", "stream_id": "employee-4411", "zone_id": "Juan.Perez"}]})
    assert is_hash(out["stream_id"]) and is_hash(out["zone_id"])
    assert is_hash(out["evidence"][0]["stream_id"]) and is_hash(out["evidence"][0]["zone_id"])
    env = f.envelope({"source": "Juan.Perez", "stream_id": "employee-4411", "payload": {}})
    assert is_hash(env["source"]) and is_hash(env["stream_id"])
    assert "employee" not in dump([out, env]) and "Juan" not in dump([out, env])


def test_allowlist_explicita_deja_pasar_camaras_y_zonas_configuradas():
    f = flt(allowed_streams=["cam-1"], allowed_zones=["bodega"])
    out = f.payload({"stream_id": "cam-1", "zone_id": "bodega"})
    assert out == {"stream_id": "cam-1", "zone_id": "bodega"}
    assert is_hash(f.payload({"stream_id": "cam-2"})["stream_id"])
    assert is_hash(f.payload({"stream_id": "cam-1\n"})["stream_id"])  # sin \n al final


def test_allowlist_desde_entorno():
    f = PrivacyFilter.from_env(
        {"TRACING_ALLOWED_STREAMS": "cam-1, cam-2", "TRACING_ALLOWED_ZONES": "bodega"}
    )
    assert f.payload({"stream_id": "cam-2", "zone_id": "bodega"}) == {
        "stream_id": "cam-2", "zone_id": "bodega"
    }


def test_sources_de_agentes_conocidos_pasan():
    f = flt()
    for role in ("event", "supervisor", "fusion"):
        assert f.envelope({"source": role, "payload": {}})["source"] == role


# -- resto ------------------------------------------------------------------------


def test_allowlist_descarta_biometria_imagenes_y_ids_en_claro():
    text = dump(flt().payload(HOSTIL))
    for secreto in (
        "embedding", "data:image", "QUJD", "face_crop", "crop.jpg", "AA:BB:CC",
        "04A224B2", "Juan Perez", "bbox", "0.01", "p-7f3a", "dec-0001",
    ):
        assert secreto not in text, secreto


def test_conserva_los_campos_permitidos():
    out = flt().payload(HOSTIL)
    assert out["outcome"] == "alert" and out["confidence"] == 0.62
    assert out["reason_codes"] == ["unknown_face"] and out["requires_operator"] is True
    (ev,) = out["evidence"]
    assert ev["kind"] == "identity" and ev["role"] == "supports" and ev["confidence"] == 0.9
    assert "detail" not in ev and "embedding" not in ev


def test_fechas_solo_iso_validas_y_sin_espacios_sobrantes():
    out = flt().payload({"evaluated_at": "2026-01-01T00:00:00+00:00"})
    assert out["evaluated_at"].startswith("2026-01-01")
    for malo in ("Juan Perez", "2026-01-01T00:00:00+00:00\n", " 2026-01-01"):
        assert "evaluated_at" not in flt().payload({"evaluated_at": malo})


@pytest.mark.parametrize("malo", [math.nan, math.inf, -math.inf, 1.5, -0.1, "0.5", True])
def test_confidence_no_finita_o_fuera_de_rango_se_descarta(malo):
    assert "confidence" not in flt().payload({"confidence": malo})


def test_las_salidas_son_json_estricto():
    json.dumps(flt().payload({"confidence": math.nan, **HOSTIL}), allow_nan=False)


def test_listas_de_objetos_solo_para_detections_y_tracks():
    out = flt().payload({"detections": [{"id": 1}, {"id": 2}], "tracks": [{"id": 1}]})
    assert out["detections_count"] == 2 and out["tracks_count"] == 1
    assert "juan_perez" not in dump(flt().payload({"juan_perez": [{"id": 1}]}))


def test_deny_by_default_cuenta_campos_descartados_sin_contenido():
    f = flt()
    out = f.payload({"type": "loitering", "embedding": [1], "name": "Juan", "tag_id": "t"})
    assert out["redacted_fields"] == 3 and f.redacted_fields == 3
    assert "Juan" not in dump(out)
    assert "redacted_fields" not in f.payload({"type": "loitering"})


def test_envelope_solo_campos_de_trazabilidad():
    out = flt().envelope(
        {"topic": "events", "source": "fusion", "payload": HOSTIL, "stream_id": "cam-1",
         "created_at": "2026-01-01T00:00:00+00:00", "event_id": "employee-4411@corp",
         "correlation_id": "co@rp", "causation_id": None, "schema_version": 1}
    )
    assert set(out) == {
        "topic", "source", "stream_id", "created_at", "event_id", "correlation_id",
        "causation_id", "payload",
    }
    assert out["causation_id"] is None and out["topic"] == "events"


def test_hop_filtrado():
    out = flt().hop(
        {"event_id": "a", "topic": "events", "source": "event", "stream_id": "cam-1",
         "causation_id": "b", "created_at": "2026-01-01T00:00:00+00:00",
         "hop_latency_ms": math.nan, "payload": {"x": 1}, "parent_event_ids": ["a@b"]}
    )
    assert out["hop_latency_ms"] is None and "payload" not in out


# -- totalidad: nunca lanza -----------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [None, 5, "texto", [1], {"evidence": [{"evidence_id": "e", "kind": ["x"], "role": {"a": 1}}]},
     {"outcome": ["a"], "type": {"a": 1}, "reason_codes": [["a"], {"b": 1}, None]},
     {"evidence": "no-lista"}, {"evidence": [None, 3, "x"]}, {"person_id": ["x"], "track_ref": {"a": 1}},
     {"confidence": object()}, {"count": 2**100}, {1: "clave-no-texto"}],
)
def test_payload_nunca_lanza(payload):
    out = flt().payload(payload)
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize(
    "data",
    [None, {}, {"topic": ["a"], "source": ["b"], "payload": None, "event_id": ["x"]},
     {"topic": 1, "stream_id": {"a": 1}, "created_at": [1], "causation_id": 5}],
)
def test_envelope_nunca_lanza(data):
    json.dumps(flt().envelope(data), allow_nan=False)


@pytest.mark.parametrize(
    "hop",
    [None, {}, {"topic": [1], "source": {"a": 1}, "event_id": ["x"], "parent_event_ids": 5},
     {"parent_event_ids": [["a"], {"b": 1}, None], "hop_latency_ms": "no"}],
)
def test_hop_nunca_lanza(hop):
    json.dumps(flt().hop(hop), allow_nan=False)
