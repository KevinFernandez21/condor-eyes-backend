"""Búsqueda por embeddings (issue #27): embedders, índice vectorial, identidad, enrolamiento
y benchmark. Todo con embedders falsos: sin red, sin cámara, sin pesos."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from faceid import VerifyStatus
from faceid.embedders import (
    CloudConsentError,
    EmbedderEngine,
    GeminiEmbedder,
    GeminiRateLimitError,
)
from faceid.enroll import EnrollError, enroll_person
from faceid.enroll_cli import main as enroll_main
from faceid.identity import IdentityHandler, IdentityPolicy, json_safe
from faceid.vectorstore import FaceIndex, decide_open_set, index_path

KEY = "AIzaSy-SECRET-KEY-123"


def unit(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype="float32")
    return v / np.linalg.norm(v)


@dataclass
class FakeFace:
    emb: np.ndarray
    box: tuple[float, float, float, float] = (10, 10, 80, 80)
    score: float = 0.95
    yaw_ratio: float = 0.0

    @property
    def size(self) -> float:
        return min(self.box[2], self.box[3])


class FakeDetector:
    """El píxel (0,0,0) elige el escenario: 0 = sin cara, otro valor = índice de persona."""

    def __init__(self, people: dict[int, np.ndarray]) -> None:
        self.people = people

    def detect(self, img):
        tag = int(img[0, 0, 0])
        if tag not in self.people:
            return img, []
        return img, [FakeFace(self.people[tag])]


class FakeEmbedder:
    def __init__(self, model_id: str = "fake", cloud: bool = False) -> None:
        self.model_id = model_id
        self.cloud = cloud
        self.batches: list[int] = []

    def embed(self, img, face):
        return face.emb

    def embed_many(self, items):
        self.batches.append(len(items))
        return [f.emb for _, f in items]


def frame(tag: int) -> np.ndarray:
    img = (np.random.default_rng(1).random((120, 120, 3)) * 255).astype("uint8")
    img[0, 0, 0] = tag
    return img


PEOPLE = {1: unit(1, 0.05, 0), 2: unit(0, 0, 1), 3: unit(0, 1, 0)}


# --------------------------------------------------------------------- Gemini


class FakeModels:
    def __init__(self, fail_429: int = 0, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_429 = fail_429
        self.error = error

    def embed_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "n": len(contents), "config": config})
        if self.fail_429 > 0:
            self.fail_429 -= 1
            raise RateErr()
        if self.error:
            raise self.error
        dim = (config or {}).get("output_dimensionality", 4)
        vecs = [
            type("E", (), {"values": [float(i + 1)] + [0.5] * (dim - 1)})()
            for i in range(len(contents))
        ]
        return type("R", (), {"embeddings": vecs})()


class RateErr(Exception):
    code = 429

    def __str__(self) -> str:
        return f"429 RESOURCE_EXHAUSTED key={KEY}"


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models


def items(n: int) -> list[tuple[np.ndarray, FakeFace]]:
    return [(frame(1), FakeFace(unit(1, 0, 0))) for _ in range(n)]


def gemini(models: FakeModels, **kw: Any) -> tuple[GeminiEmbedder, list[float]]:
    sleeps: list[float] = []
    g = GeminiEmbedder(
        cloud_consent=kw.pop("cloud_consent", True),
        client=FakeClient(models),
        sleep=sleeps.append,
        dim=8,
        **kw,
    )
    return g, sleeps


def test_gemini_refuses_without_cloud_consent():
    models = FakeModels()
    g, _ = gemini(models, cloud_consent=False)
    with pytest.raises(CloudConsentError):
        g.embed_many(items(2))
    assert models.calls == []  # nada salió hacia la nube


def test_gemini_api_key_only_from_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    g = GeminiEmbedder(cloud_consent=True)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        g.embed_many(items(1))


def test_gemini_batches_and_normalizes():
    models = FakeModels()
    g, _ = gemini(models, batch_size=2)
    out = g.embed_many(items(5))
    assert [c["n"] for c in models.calls] == [2, 2, 1]
    assert len(out) == 5
    assert all(math.isclose(float(np.linalg.norm(v)), 1.0, rel_tol=1e-5) for v in out)
    assert models.calls[0]["model"] == "gemini-embedding-2"
    assert models.calls[0]["config"]["output_dimensionality"] == 8
    assert g.model_id == "gemini-embedding-2@8" and g.cloud is True


def test_gemini_backoff_on_429_then_succeeds():
    models = FakeModels(fail_429=2)
    g, sleeps = gemini(models, base_delay=1.0)
    out = g.embed_many(items(1))
    assert len(out) == 1
    assert len(models.calls) == 3
    assert sleeps == [1.0, 2.0]  # exponencial


def test_gemini_gives_up_and_never_leaks_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    models = FakeModels(fail_429=99)
    sleeps: list[float] = []
    g = GeminiEmbedder(
        cloud_consent=True,
        client=FakeClient(models),
        sleep=sleeps.append,
        max_retries=3,
        dim=8,
    )
    with pytest.raises(GeminiRateLimitError) as e:
        g.embed_many(items(1))
    assert KEY not in str(e.value)
    assert len(models.calls) == 4  # 1 intento + 3 reintentos


def test_gemini_other_errors_are_not_retried_and_redacted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    models = FakeModels(error=ValueError(f"bad request {KEY}"))
    g, sleeps = gemini(models)
    with pytest.raises(RuntimeError) as e:
        g.embed_many(items(1))
    assert KEY not in str(e.value)
    assert len(models.calls) == 1 and sleeps == []


# ------------------------------------------------------------- índice vectorial


def make_index(tmp_path: Path, model: str = "fake", cloud: bool = False) -> FaceIndex:
    return FaceIndex(tmp_path / "idx.sqlite", model_id=model, cloud=cloud)


def test_index_search_returns_best_per_person(tmp_path):
    idx = make_index(tmp_path)
    idx.add("A", [unit(1, 0, 0), unit(0.9, 0.1, 0)], consent_ref="C-1")
    idx.add("B", [unit(0, 1, 0)], consent_ref="C-2")
    hits = idx.search(unit(1, 0.02, 0), k=5)
    assert [h.person_id for h in hits] == ["A", "B"]  # una entrada por persona
    assert hits[0].score > 0.99 > hits[1].score


def test_index_requires_consent_and_vectors(tmp_path):
    idx = make_index(tmp_path)
    with pytest.raises(ValueError, match="consentimiento"):
        idx.add("A", [unit(1, 0, 0)], consent_ref="  ")
    with pytest.raises(ValueError):
        idx.add("A", [], consent_ref="C-1")


def test_cloud_index_requires_cloud_consent(tmp_path):
    idx = make_index(tmp_path, "gemini-embedding-2@8", cloud=True)
    with pytest.raises(CloudConsentError):
        idx.add("A", [unit(1, 0, 0)], consent_ref="C-1", cloud_consent=False)
    assert idx.count() == 0
    idx.add("A", [unit(1, 0, 0)], consent_ref="C-1", cloud_consent=True)
    assert idx.count() == 1


def test_index_delete_and_purge_and_expiry(tmp_path):
    idx = make_index(tmp_path)
    idx.add("A", [unit(1, 0, 0)], consent_ref="C-1", retention_days=1, now=1000.0)
    idx.add("B", [unit(0, 1, 0)], consent_ref="C-2", retention_days=30, now=1000.0)
    later = 1000.0 + 2 * 86400
    assert [h.person_id for h in idx.search(unit(1, 0, 0), now=later)] == ["B"]
    assert idx.purge_expired(now=later) == ["A"]
    assert idx.delete("B") is True
    assert idx.delete("B") is False
    assert idx.count() == 0


def test_index_persists_only_vectors_and_refs(tmp_path):
    idx = make_index(tmp_path)
    idx.add("A", [unit(1, 0, 0)], consent_ref="C-1")
    idx.close()
    con = sqlite3.connect(tmp_path / "idx.sqlite")
    cols = {r[1] for r in con.execute("PRAGMA table_info(vectors)")}
    con.close()
    assert cols == {
        "id",
        "person_id",
        "vec",
        "consent_ref",
        "cloud_consent",
        "enrolled_at",
        "expires_at",
    }
    reopened = make_index(tmp_path)
    assert reopened.count() == 1  # persiste


def test_index_rejects_other_model_and_dim(tmp_path):
    make_index(tmp_path, "model-a").add("A", [unit(1, 0, 0)], consent_ref="C-1")
    with pytest.raises(ValueError, match="model-a"):
        make_index(tmp_path, "model-b")
    idx = make_index(tmp_path, "model-a")
    with pytest.raises(ValueError, match="imensi"):
        idx.add("B", [unit(1, 0)], consent_ref="C-2")


def test_index_path_is_per_embedder(tmp_path):
    a = index_path(tmp_path, "sface")
    b = index_path(tmp_path, "gemini-embedding-2@768")
    assert a != b and ":" not in b.name and "@" not in b.name


def test_audit_has_no_biometrics(tmp_path):
    idx = make_index(tmp_path)
    idx.add("A", [unit(1, 0.123456, 0)], consent_ref="C-1", actor="kevin")
    idx.delete("A", actor="kevin")
    log = (tmp_path / "idx.audit.jsonl").read_text(encoding="utf-8")
    actions = [json.loads(line)["action"] for line in log.splitlines()]
    assert actions == ["enroll", "delete"]
    assert "0.123456" not in log and "vec" not in log


def test_decide_open_set():
    from faceid.vectorstore import Hit

    thr, mar = 0.6, 0.05
    assert decide_open_set([], thr, mar)[0] is VerifyStatus.UNKNOWN
    assert decide_open_set([Hit("A", 0.4)], thr, mar)[0] is VerifyStatus.UNKNOWN
    st, who, *_ = decide_open_set([Hit("A", 0.9), Hit("B", 0.2)], thr, mar)
    assert st is VerifyStatus.MATCH and who == "A"
    st, who, *_ = decide_open_set([Hit("A", 0.9), Hit("B", 0.88)], thr, mar)
    assert st is VerifyStatus.INCONCLUSIVE and who is None


# ------------------------------------------------------------------- identidad


def handler(tmp_path: Path, **kw: Any) -> tuple[IdentityHandler, FaceIndex]:
    emb = kw.pop("embedder", FakeEmbedder())
    idx = FaceIndex(tmp_path / "i.sqlite", emb.model_id, cloud=emb.cloud)
    idx.add("EMP-1", [PEOPLE[1]], consent_ref="C-1", cloud_consent=emb.cloud)
    engine = EmbedderEngine(FakeDetector(PEOPLE), emb)
    return IdentityHandler(engine, emb, idx, IdentityPolicy(0.6, 0.05), **kw), idx


def assert_strict_json(payload: Any) -> None:
    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                assert isinstance(k, str)
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        else:
            assert v is None or type(v) in (str, int, float, bool), type(v)
            if isinstance(v, float):
                assert math.isfinite(v)

    walk(payload)
    json.dumps(payload, allow_nan=False)


def test_identity_match_envelope_is_metadata_only(tmp_path):
    h, _ = handler(tmp_path)
    env = h.identify(frame(1), stream_id="cam-1", now=1_700_000_000.0)
    p = env.payload
    assert env.source == "identity" and env.stream_id == "cam-1"
    assert p["status"] == "match" and p["person_id"] == "EMP-1"
    assert p["requires_operator"] is True and p["model_id"] == "fake"
    assert p["observed_at"] == "2023-11-14T22:13:20+00:00"
    assert_strict_json(dict(p))
    assert not any(k in p for k in ("embedding", "vector", "image", "frame", "allow"))


def test_identity_unknown_no_face_and_invalid(tmp_path):
    h, _ = handler(tmp_path)
    assert h.identify(frame(2)).payload["status"] == "unknown"
    assert h.identify(frame(9)).payload["status"] == "no_face"
    assert h.identify(None).payload["status"] == "invalid_input"  # type: ignore[arg-type]
    for t in (2, 9):
        p = h.identify(frame(t)).payload
        assert p["person_id"] is None and p["requires_operator"] is True
        assert_strict_json(dict(p))


def test_identity_blocks_cloud_embedder_unless_allowed(tmp_path):
    emb = FakeEmbedder("gemini-embedding-2@8", cloud=True)
    idx = FaceIndex(tmp_path / "c.sqlite", emb.model_id, cloud=True)
    engine = EmbedderEngine(FakeDetector(PEOPLE), emb)
    with pytest.raises(CloudConsentError):
        IdentityHandler(engine, emb, idx)
    IdentityHandler(engine, emb, idx, allow_cloud=True)


def test_identity_rejects_index_of_other_model(tmp_path):
    emb = FakeEmbedder("a")
    idx = FaceIndex(tmp_path / "x.sqlite", "b")
    with pytest.raises(ValueError):
        IdentityHandler(EmbedderEngine(FakeDetector(PEOPLE), emb), emb, idx)


def test_json_safe_sanitizes():
    out = json_safe(
        {
            "a": np.float32(0.5),
            "b": float("nan"),
            "c": np.int64(3),
            "d": [np.float64(1)],
        }
    )
    assert out == {"a": 0.5, "b": None, "c": 3, "d": [1.0]}
    assert_strict_json(out)


# ----------------------------------------------------------------- enrolamiento


def test_enroll_person_stores_vectors_and_discards_bad_frames(tmp_path):
    emb = FakeEmbedder()
    idx = FaceIndex(tmp_path / "e.sqlite", emb.model_id)
    engine = EmbedderEngine(FakeDetector(PEOPLE), emb)
    frames = iter([frame(9), frame(1), frame(1), frame(9), frame(1), frame(1)])
    rep = enroll_person(frames, engine, emb, idx, "EMP-1", "C-1", n=3, min_samples=3)
    assert rep["samples"] == 3 and rep["rejected"] == {"no_face": 2}
    assert emb.batches == [3]
    assert idx.count() == 1


def test_enroll_person_fails_without_enough_faces(tmp_path):
    emb = FakeEmbedder()
    idx = FaceIndex(tmp_path / "e.sqlite", emb.model_id)
    engine = EmbedderEngine(FakeDetector(PEOPLE), emb)
    with pytest.raises(EnrollError):
        enroll_person(iter([frame(9)] * 4), engine, emb, idx, "A", "C-1", n=3)
    assert idx.count() == 0


def test_enroll_cloud_requires_cloud_consent(tmp_path):
    emb = FakeEmbedder("gemini-embedding-2@8", cloud=True)
    idx = FaceIndex(tmp_path / "g.sqlite", emb.model_id, cloud=True)
    engine = EmbedderEngine(FakeDetector(PEOPLE), emb)
    with pytest.raises(CloudConsentError):
        enroll_person(iter([frame(1)] * 5), engine, emb, idx, "A", "C-1", n=3)
    assert emb.batches == []  # ni siquiera se calculó el embedding


def test_enroll_cli_dry_run_needs_no_camera_and_writes_nothing(tmp_path, capsys):
    rc = enroll_main(
        [
            "--dry-run",
            "--person-id",
            "EMP-1",
            "--consent-ref",
            "C-1",
            "--index-dir",
            str(tmp_path / "idx"),
        ]
    )
    assert rc == 0
    assert not (tmp_path / "idx").exists()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["dry_run"] is True and out["samples"] >= 3


def test_enroll_cli_requires_consent_ref(capsys):
    with pytest.raises(SystemExit):
        enroll_main(["--dry-run", "--person-id", "A"])


def test_enroll_cli_gemini_requires_cloud_consent(tmp_path, capsys):
    rc = enroll_main(
        [
            "--dry-run",
            "--embedder",
            "gemini",
            "--person-id",
            "A",
            "--consent-ref",
            "C-1",
        ]
    )
    assert rc != 0


# -------------------------------------------------------------------- benchmark


class SubjectDetector:
    """Lee el sujeto del píxel (0,0,0) del PNG sintético."""

    def detect(self, img):
        return img, [FakeFace(np.zeros(3, dtype="float32"), box=(0, 0, 100, 100))]


class SubjectEmbedder:
    cloud = False

    def __init__(self, noise: float, model_id: str = "subj") -> None:
        self.model_id = model_id
        self.noise = noise
        self.rng = np.random.default_rng(0)

    def embed(self, img, face):
        return self.embed_many([(img, face)])[0]

    def embed_many(self, items):
        out = []
        for img, _ in items:
            s = int(img[0, 0, 0])
            base = np.random.default_rng(1000 + s).normal(size=16)
            v = base + self.rng.normal(size=16) * self.noise
            out.append((v / np.linalg.norm(v)).astype("float32"))
        return out


def write_subjects(root: Path, n_subjects: int, per: int) -> None:
    for s in range(n_subjects):
        (root / str(s)).mkdir(parents=True)
        for i in range(per):
            img = np.full((120, 120, 3), 128, dtype="uint8")
            img[0, 0, 0] = s
            img[1:, 1:] = np.random.default_rng(s * 100 + i).random((119, 119, 3)) * 255
            assert cv2.imwrite(str(root / str(s) / f"{i}.png"), img)


def test_embedder_benchmark_reports_rates_and_cost(tmp_path):
    from faceid.embed_benchmark import run_embedder

    write_subjects(tmp_path, 12, 8)
    rep = run_embedder(
        EmbedderEngine(SubjectDetector(), SubjectEmbedder(0.05)),
        SubjectEmbedder(0.05),
        tmp_path,
        val_subjects=list(range(6)),
        test_subjects=list(range(6, 12)),
        enroll_images=range(5),
        probe_images=range(5, 8),
        target_far=0.01,
        price_per_image_usd=0.001,
    )
    assert rep["status"] == "ejecutado"
    assert rep["test"]["far"] <= 0.05 and rep["test"]["frr"] <= 0.2
    assert rep["images_embedded"] > 0
    assert rep["cost_estimate_usd"] == pytest.approx(rep["images_embedded"] * 0.001)
    assert rep["latency"]["p50_single_ms"] >= 0


def test_embedder_benchmark_rejects_identity_leakage(tmp_path):
    from faceid.embed_benchmark import run_embedder

    e = SubjectEmbedder(0.1)
    with pytest.raises(AssertionError, match="fuga"):
        run_embedder(
            EmbedderEngine(SubjectDetector(), e),
            e,
            tmp_path,
            val_subjects=[0, 1],
            test_subjects=[1, 2],
            enroll_images=range(5),
            probe_images=range(5, 8),
            target_far=0.01,
        )


def test_gemini_part_is_skipped_without_key(monkeypatch):
    from faceid.embed_benchmark import skipped_report

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = skipped_report("gemini-embedding-2@768", "GEMINI_API_KEY no definida")
    assert r["status"] == "no ejecutado" and "far" not in r
