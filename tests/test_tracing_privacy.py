"""Filtro de privacidad previo a Langfuse: deny-by-default, sin biometría ni IDs en claro."""

import json
import math

import pytest

from tracing.privacy import PrivacyFilter, sanitize_envelope, sanitize_payload

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


def dump(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def flt() -> PrivacyFilter:
    return PrivacyFilter(KEY)


def test_allowlist_descarta_biometria_imagenes_y_ids_en_claro():
    out = flt().payload(HOSTIL)
    text = dump(out)
    for secreto in (
        "embedding", "data:image", "QUJD", "face_crop", "crop.jpg", "AA:BB:CC",
        "04A224B2", "Juan Perez", "bbox", "0.01", "p-7f3a",
    ):
        assert secreto not in text, secreto


def test_conserva_los_campos_permitidos():
    out = flt().payload(HOSTIL)
    assert out["decision_id"] == "dec-0001"
    assert out["outcome"] == "alert"
    assert out["confidence"] == 0.62
    assert out["reason_codes"] == ["unknown_face"]
    assert out["zone_id"] == "bodega" and out["stream_id"] == "cam-1"
    assert out["requires_operator"] is True
    (ev,) = out["evidence"]
    assert ev["kind"] == "identity" and ev["role"] == "supports"
    assert ev["confidence"] == 0.9 and ev["zone_id"] == "bodega"
    assert "detail" not in ev and "embedding" not in ev


def test_person_id_nunca_sale_en_claro_ni_con_forma_de_seudonimo():
    for valor in ("p-7f3a", "anon-cafe", "Juan Perez", "12345678", ""):
        out = flt().payload({"person_id": valor})
        assert out["person_id"].startswith("h-")
        assert valor not in out["person_id"] or valor == ""
        assert len(out["person_id"]) == 2 + 16


def test_el_hash_es_estable_con_la_misma_clave_y_distinto_con_otra():
    a = PrivacyFilter(KEY).payload({"person_id": "p-1234"})["person_id"]
    b = PrivacyFilter(KEY).payload({"person_id": "p-1234"})["person_id"]
    c = PrivacyFilter(b"z" * 32).payload({"person_id": "p-1234"})["person_id"]
    assert a == b and a != c


def test_sin_clave_se_genera_una_aleatoria_por_proceso():
    a = PrivacyFilter().payload({"person_id": "p-1"})["person_id"]
    b = PrivacyFilter().payload({"person_id": "p-1"})["person_id"]
    assert a != b


def test_reason_codes_solo_vocabulario_cerrado_no_floats():
    out = flt().payload({"reason_codes": [0.1] * 512})
    assert "reason_codes" not in out
    out = flt().payload({"reason_codes": ["unknown_face", "made_up_code", 3, "A" * 80, "x@y"]})
    assert out["reason_codes"] == ["unknown_face"]


@pytest.mark.parametrize("campo", ["type", "action", "target", "state", "stage"])
def test_identificadores_cortos_por_campo(campo):
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
    for malo in (png, "employee-4411@corp", "Data:xxx", "A" * 33, "UPPER", "a b", 7, ["a"]):
        assert campo not in flt().payload({campo: malo}), malo
    assert flt().payload({campo: "person_in_zone"})[campo] == "person_in_zone"


def test_outcome_y_kind_son_vocabulario_cerrado():
    out = flt().payload({"outcome": "juan_perez_4411", "evidence": [
        {"evidence_id": "e1", "kind": "juan", "role": "supports"}]})
    assert "outcome" not in out
    assert out["evidence"][0].get("kind") is None


def test_evidence_id_y_campos_de_id_no_confiables_se_hashean():
    out = flt().payload(
        {"evidence": [{"evidence_id": "employee-4411@corp", "kind": "track", "role": "context"}],
         "decision_id": "employee-4411@corp", "track_ref": "Juan", "stream_id": "a@b",
         "zone_id": "x" * 200}
    )
    text = dump(out)
    assert "employee" not in text and "Juan" not in text and "a@b" not in text
    assert out["evidence"][0]["evidence_id"].startswith("h-")
    assert out["decision_id"].startswith("h-") and out["track_ref"].startswith("h-")
    assert out["stream_id"].startswith("h-") and out["zone_id"].startswith("h-")


def test_ids_de_sistema_con_formato_conocido_pasan_tal_cual():
    ev = "0a2093bbd06f4a3db2be4e4e49dd6727/tracks"
    out = flt().payload({"evidence": [{"evidence_id": ev, "kind": "track", "role": "supports"}]})
    assert out["evidence"][0]["evidence_id"] == ev


def test_fechas_solo_iso_validas():
    out = flt().payload({"evaluated_at": "2026-01-01T00:00:00+00:00"})
    assert out["evaluated_at"].startswith("2026-01-01")
    assert "evaluated_at" not in flt().payload({"evaluated_at": "Juan Perez"})


@pytest.mark.parametrize("malo", [math.nan, math.inf, -math.inf, 1.5, -0.1, "0.5", True])
def test_confidence_no_finita_o_fuera_de_rango_se_descarta(malo):
    assert "confidence" not in flt().payload({"confidence": malo})


def test_las_salidas_son_json_estricto():
    out = flt().payload({"confidence": math.nan, **HOSTIL})
    json.dumps(out, allow_nan=False)


def test_listas_de_objetos_solo_para_detections_y_tracks():
    out = flt().payload({"detections": [{"id": 1}, {"id": 2}], "tracks": [{"id": 1}]})
    assert out["detections_count"] == 2 and out["tracks_count"] == 1
    out = flt().payload({"juan_perez": [{"id": 1}]})
    assert "juan_perez" not in dump(out) and "juan_perez_count" not in dump(out)


def test_deny_by_default_cuenta_campos_descartados_sin_contenido():
    f = flt()
    out = f.payload({"type": "x", "embedding": [1], "name": "Juan", "tag_id": "t"})
    assert out["redacted_fields"] == 3
    assert f.redacted_fields == 3
    assert "Juan" not in dump(out)
    assert "redacted_fields" not in f.payload({"type": "x"})


def test_sanitize_envelope_solo_campos_de_trazabilidad_y_ids_validados():
    data = {
        "topic": "events", "source": "fusion", "payload": HOSTIL, "stream_id": "cam-1",
        "created_at": "2026-01-01T00:00:00+00:00", "schema_version": 1, "payload_version": 1,
        "event_id": "employee-4411@corp", "correlation_id": "co@rp", "causation_id": None,
    }
    out = flt().envelope(data)
    assert set(out) == {
        "topic", "source", "stream_id", "created_at", "event_id", "correlation_id",
        "causation_id", "payload",
    }
    text = dump(out)
    assert "employee" not in text and "co@rp" not in text and "embedding" not in text
    assert out["event_id"].startswith("h-") and out["causation_id"] is None
    assert out["topic"] == "events"


def test_envelope_con_topic_o_source_hostil():
    out = flt().envelope(
        {"topic": "Juan", "source": "Juan Perez <jp@corp>", "payload": {}, "event_id": "a" * 32,
         "created_at": "no-es-fecha"}
    )
    text = dump(out)
    assert "Juan" not in text and "corp" not in text and "no-es-fecha" not in text


def test_hop_filtrado():
    hop = {
        "event_id": "employee-4411@corp", "topic": "events", "source": "event",
        "stream_id": "cam-1", "causation_id": "0a" * 16, "created_at": "2026-01-01T00:00:00+00:00",
        "hop_latency_ms": math.nan, "payload": {"x": 1}, "parent_event_ids": ["a@b"],
    }
    out = flt().hop(hop)
    text = dump(out)
    assert "employee" not in text and "a@b" not in text and "payload" not in text
    assert out["hop_latency_ms"] is None


def test_wrappers_de_modulo():
    assert sanitize_payload({"type": "x", "embedding": [1]})["type"] == "x"
    assert sanitize_envelope({"payload": {}, "event_id": "a" * 32})["event_id"] == "a" * 32
