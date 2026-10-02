"""Filtro de privacidad previo a Langfuse: allowlist, sin biometría ni IDs en claro."""

import json

import pytest

from tracing.privacy import is_pseudonym, sanitize_envelope, sanitize_payload

HOSTIL = {
    "decision_id": "dec-0001",
    "outcome": "uncorroborated",
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
    return json.dumps(value, sort_keys=True)


def test_allowlist_descarta_biometria_imagenes_y_ids_en_claro():
    out = sanitize_payload(HOSTIL)
    text = dump(out)
    for secreto in (
        "embedding", "data:image", "QUJD", "face_crop", "crop.jpg", "AA:BB:CC",
        "04A224B2", "Juan Perez", "bbox", "0.01",
    ):
        assert secreto not in text, secreto


def test_conserva_los_campos_permitidos():
    out = sanitize_payload(HOSTIL)
    assert out["decision_id"] == "dec-0001"
    assert out["outcome"] == "uncorroborated"
    assert out["confidence"] == 0.62
    assert out["reason_codes"] == ["unknown_face"]
    assert out["zone_id"] == "bodega" and out["stream_id"] == "cam-1"
    assert out["person_id"] == "p-7f3a"  # seudónimo: permitido
    assert out["requires_operator"] is True
    assert out["evidence"] == [
        {"evidence_id": "e1", "kind": "identity", "role": "supports",
         "confidence": 0.9, "zone_id": "bodega"}
    ]


def test_las_listas_grandes_solo_dejan_su_conteo():
    out = sanitize_payload({"type": "x", "detections": [{"id": 1}, {"id": 2}]})
    assert out == {"type": "x", "detections_count": 2}


@pytest.mark.parametrize("valor", ["Juan Perez", "12345678", "AA:BB:CC:DD:EE:FF", "p-zz", ""])
def test_person_id_no_seudonimizado_se_redacta(valor):
    out = sanitize_payload({"person_id": valor})
    assert out == {"person_id": "[redacted]"}


@pytest.mark.parametrize("valor", ["p-7f3a", "p-0123456789abcdef", "anon-ab12cd34"])
def test_is_pseudonym_acepta_formas_hash(valor):
    assert is_pseudonym(valor)


def test_valores_permitidos_pero_peligrosos_se_descartan():
    out = sanitize_payload(
        {"type": "data:image/png;base64,AAAA", "zone_id": "z" * 500, "outcome": ["a", {"b": 1}]}
    )
    assert "type" not in out
    assert len(out["zone_id"]) <= 120
    assert "outcome" not in out  # estructura inesperada en un campo escalar


def test_sanitize_envelope_solo_deja_campos_de_trazabilidad():
    data = {
        "topic": "events", "source": "fusion", "payload": HOSTIL, "stream_id": "cam-1",
        "created_at": "2026-01-01T00:00:00+00:00", "schema_version": 1, "payload_version": 1,
        "event_id": "ev-1", "correlation_id": "co-1", "causation_id": None,
    }
    out = sanitize_envelope(data)
    assert set(out) == {
        "topic", "source", "stream_id", "created_at", "event_id", "correlation_id",
        "causation_id", "payload",
    }
    assert "embedding" not in dump(out)
