"""Tests de verificación facial: match, no-match, sin cara, varias caras, entrada corrupta,
calidad baja, consentimiento, retención, auditoría y contrato de metadata."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from faceid import EnrollmentStore, Verifier, VerifyPolicy, VerifyStatus
from faceid.benchmark import Probe, calibrate, decide, rates, split_subjects, templates
from faceid.digiface import HttpRangeFile


def unit(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype="float32")
    return v / np.linalg.norm(v)


@dataclass
class FakeFace:
    box: tuple[float, float, float, float] = (10, 10, 80, 80)
    score: float = 0.95
    yaw_ratio: float = 0.0
    emb: np.ndarray = field(default_factory=lambda: unit(1, 0, 0))

    @property
    def size(self) -> float:
        return min(self.box[2], self.box[3])


class FakeEngine:
    """El valor del píxel (0,0,0) elige el escenario; así no hacen falta modelos."""

    def __init__(self, faces: dict[int, list[FakeFace]]) -> None:
        self.faces = faces

    def detect(self, img):
        return img, self.faces.get(int(img[0, 0, 0]), [])

    def embed(self, img, face):
        return face.emb


def textured(tag: int) -> np.ndarray:
    img = (np.random.default_rng(0).random((120, 120, 3)) * 255).astype("uint8")
    img[0, 0, 0] = tag
    return img


SCEN = {
    1: [FakeFace(emb=unit(1, 0.05, 0))],  # persona A
    2: [FakeFace(emb=unit(0, 0, 1))],  # desconocido
    3: [],  # sin cara
    4: [FakeFace(), FakeFace()],  # dos caras
    5: [FakeFace(box=(0, 0, 20, 20))],  # cara pequeña
    6: [FakeFace(yaw_ratio=0.6)],  # perfil
    7: [FakeFace(emb=unit(0.7, 0.7, 0))],  # equidistante de A y B
}


@pytest.fixture
def setup(tmp_path: Path):
    store = EnrollmentStore(tmp_path / "enrolled.json")
    store.enroll("A", [unit(1, 0, 0), unit(1, 0.1, 0)], consent_ref="CONS-001", now=0)
    store.enroll("B", [unit(0, 1, 0)], consent_ref="CONS-002", now=0)
    return Verifier(
        FakeEngine(SCEN),
        store,
        VerifyPolicy(threshold=0.6, margin=0.05, min_sharpness=1.0),
    ), store


def test_match(setup):
    v, _ = setup
    r = v.verify(textured(1), now=1)
    assert r.status is VerifyStatus.MATCH and r.person_id == "A" and r.score > 0.9


def test_non_match_is_unknown(setup):
    v, _ = setup
    r = v.verify(textured(2), now=1)
    assert r.status is VerifyStatus.UNKNOWN and r.person_id is None


def test_missing_face(setup):
    assert setup[0].verify(textured(3), now=1).status is VerifyStatus.NO_FACE


def test_multiple_faces_is_explicit(setup):
    r = setup[0].verify(textured(4), now=1)
    assert r.status is VerifyStatus.MULTIPLE_FACES and r.evidence["faces"] == 2


@pytest.mark.parametrize(
    "bad",
    [None, b"\x00\x01", np.zeros((5, 5, 3), "uint8"), np.zeros((50, 50), "uint8")],
)
def test_corrupt_input(setup, bad):
    assert setup[0].verify(bad, now=1).status is VerifyStatus.INVALID_INPUT


@pytest.mark.parametrize(("tag", "issue"), [(5, "small_face"), (6, "pose")])
def test_low_quality_is_inconclusive(setup, tag, issue):
    r = setup[0].verify(textured(tag), now=1)
    assert (
        r.status is VerifyStatus.INCONCLUSIVE and issue in r.evidence["quality_issues"]
    )


def test_blurry_face_is_inconclusive(tmp_path: Path):
    store = EnrollmentStore(tmp_path / "s.json")
    store.enroll("A", [unit(1, 0, 0)], consent_ref="C", now=0)
    v = Verifier(FakeEngine(SCEN), store, VerifyPolicy(min_sharpness=1e9))
    assert "blurry" in v.verify(textured(1), now=1).evidence["quality_issues"]


def test_ambiguous_between_two_enrolled_is_inconclusive(setup):
    r = setup[0].verify(textured(7), now=1)
    assert r.status is VerifyStatus.INCONCLUSIVE and r.evidence["reason"] == "ambiguous"


def test_payload_is_metadata_and_leaves_decision_to_operator(setup):
    env = setup[0].verify(textured(1), now=1).to_envelope(stream_id="cam1")
    p = json.loads(json.dumps(env.payload))
    assert p["requires_operator"] is True
    assert not {"access", "granted", "allow", "deny"} & set(p)
    assert env.source == "identity"


def test_store_keeps_only_embeddings_with_consent_and_audit(setup, tmp_path: Path):
    _, store = setup
    raw = json.loads((tmp_path / "enrolled.json").read_text())
    assert set(raw["A"]) == {
        "template",
        "samples",
        "consent_ref",
        "enrolled_at",
        "expires_at",
    }
    with pytest.raises(ValueError):
        store.enroll("C", [unit(1, 0, 0)], consent_ref=" ")
    assert store.delete("B") and "B" not in store
    actions = [
        json.loads(ln)["action"]
        for ln in (tmp_path / "enrolled.audit.jsonl").read_text().splitlines()
    ]
    assert actions == ["enroll", "enroll", "delete"]


def test_retention_expires_and_purges(tmp_path: Path):
    store = EnrollmentStore(tmp_path / "s.json")
    store.enroll("A", [unit(1, 0, 0)], consent_ref="C", retention_days=1, now=0)
    assert store.templates(now=10)[0] == ["A"]
    assert store.templates(now=2 * 86400)[0] == []
    assert store.purge_expired(now=2 * 86400) == ["A"] and "A" not in store


# --- benchmark ---------------------------------------------------------------------


def probe(subject: int, *xs: float, status=None) -> Probe:
    return Probe(subject, status, None if status else unit(*xs), {})


def test_split_subjects_disjoint():
    enr, unk = split_subjects(list(range(10)))
    assert not set(enr) & set(unk) and len(enr) == 5


def test_far_frr_counts_unknowns_and_misidentification():
    ids, mat = templates([probe(1, 1, 0, 0), probe(2, 0, 1, 0)])
    genuine = [
        probe(1, 1, 0.1, 0),
        probe(2, 1, 0, 0),
        probe(2, status=VerifyStatus.NO_FACE),
    ]
    unknown = [probe(9, 0, 0, 1), probe(8, 0.9, 0.1, 0)]
    r = rates(genuine, unknown, ids, mat, thr=0.5, margin=0.0)
    # FA: sonda de 2 aceptada como 1 + desconocido 8 aceptado → 2 / 5.
    assert r["far"] == pytest.approx(2 / 5)
    assert r["frr"] == pytest.approx(2 / 3) and r["genuine_no_face"] == pytest.approx(
        1 / 3
    )
    assert decide(unknown[0], ids, mat, 0.5, 0.0) == ("unknown", None)


def test_calibration_picks_lowest_threshold_meeting_far():
    ids, mat = templates([probe(1, 1, 0), probe(2, 0, 1)])
    genuine = [probe(1, 1, 0.2), probe(2, 0.2, 1)]
    unknown = [probe(9, 1, 1)]  # coseno ≈ 0,71 con ambos
    r = calibrate(genuine, unknown, ids, mat, target_far=0.0, margin=0.0)
    assert r["far"] == 0.0 and 0.70 < r["threshold"] <= 0.98


# --- descarga parcial ----------------------------------------------------------------


def test_http_range_file_reads_zip_members(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("0/0.png", b"abc")
        z.writestr("1/3.png", b"xyz" * 10)
    data = buf.getvalue()

    class Resp(io.BytesIO):
        headers: ClassVar[dict[str, str]] = {"Content-Length": str(len(data))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req):
        rng = req.headers.get("Range")
        if rng:
            a, b = (int(x) for x in rng.split("=")[1].split("-"))
            return Resp(data[a : b + 1])
        return Resp(b"")

    monkeypatch.setattr("faceid.digiface.urllib.request.urlopen", fake_urlopen)
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile("http://x/y.zip"), 64))
    assert z.read("1/3.png") == b"xyz" * 10


def test_payload_coerces_numpy_scalars_from_engine(setup):
    verifier, _ = setup
    verifier.engine.faces[1] = [
        FakeFace(
            box=(10, 10, 80, 80),
            score=np.float32(0.95),
            yaw_ratio=np.float32(0.0),
            emb=unit(1, 0.05, 0),
        )
    ]
    env = verifier.verify(textured(1), now=1).to_envelope(stream_id="cam1")
    json.dumps(env.payload)
    assert type(env.payload["score"]) is float
    assert type(env.payload["evidence"]["det_score"]) is float
