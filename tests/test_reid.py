"""Tests de re-ID: asociación (inconcluso, ambigüedad, duplicados, obsolescencia), splits y métricas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reid import (
    AssociatorConfig,
    CrossCameraAssociator,
    LinkStatus,
    TrackDescriptor,
    Transition,
)
from reid.benchmark import association, retrieval, tracks_from
from reid.embed import ColorEmbedder
from reid.market import Crop, build_splits, parse


def vec(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype="float32")
    return v / np.linalg.norm(v)


def td(
    stream: str, tid: int, emb: np.ndarray, t: float = 0.0, zone: str | None = None
) -> TrackDescriptor:
    return TrackDescriptor(stream, tid, t, t, emb, zone)


def assoc(**kw) -> CrossCameraAssociator:
    return CrossCameraAssociator(AssociatorConfig(threshold=0.8, margin=0.05, **kw))


# --- asociación -------------------------------------------------------------------


def test_no_candidate_when_gallery_empty_or_same_camera():
    a = assoc()
    q = td("cam2", 1, vec(1, 0), t=10)
    assert a.associate(q).status is LinkStatus.NO_CANDIDATE
    a.upsert(td("cam2", 7, vec(1, 0)))
    r = a.associate(q)
    assert r.status is LinkStatus.NO_CANDIDATE and r.evidence["rejected"] == {
        "same_camera": 1
    }


def test_clear_match_links_with_evidence():
    a = assoc()
    a.upsert(td("cam1", 1, vec(1, 0, 0)))
    a.upsert(td("cam1", 2, vec(0, 1, 0)))
    r = a.associate(td("cam2", 9, vec(0.99, 0.05, 0), t=5))
    assert r.status is LinkStatus.LINKED and r.candidate == ("cam1", 1)
    assert r.evidence["similarity"] > 0.99 and r.evidence["dt_s"] == 5
    assert 0.5 < r.confidence <= 1.0


def test_low_similarity_is_inconclusive_not_forced():
    a = assoc()
    a.upsert(td("cam1", 1, vec(1, 0)))
    r = a.associate(td("cam2", 9, vec(1, 1), t=5))
    assert (
        r.status is LinkStatus.INCONCLUSIVE and r.evidence["reason"] == "low_similarity"
    )


def test_ambiguous_candidates_are_inconclusive():
    a = assoc()
    a.upsert(td("cam1", 1, vec(1, 0.02)))
    a.upsert(td("cam3", 2, vec(1, -0.02)))
    r = a.associate(td("cam2", 9, vec(1, 0), t=5))
    assert r.status is LinkStatus.INCONCLUSIVE and r.evidence["reason"] == "ambiguous"


def test_duplicate_track_updates_instead_of_duplicating():
    a = assoc()
    a.upsert(td("cam1", 1, vec(1, 0), t=0))
    a.upsert(td("cam1", 1, vec(1, 0), t=3))
    assert len(a) == 1
    # Si se duplicara, el segundo candidato idéntico haría la consulta ambigua.
    assert a.associate(td("cam2", 9, vec(1, 0), t=5)).status is LinkStatus.LINKED


def test_stale_observations_are_pruned_and_rejected():
    a = assoc(max_age_s=60)
    a.upsert(td("cam1", 1, vec(1, 0), t=0))
    r = a.associate(td("cam2", 9, vec(1, 0), t=100))
    assert r.status is LinkStatus.NO_CANDIDATE and r.evidence["rejected"] == {
        "stale": 1
    }
    assert a.prune(now=100) == 1 and len(a) == 0


def test_overlapping_track_cannot_be_the_same_person():
    a = assoc()
    a.upsert(td("cam1", 1, vec(1, 0), t=10))
    assert a.associate(td("cam2", 9, vec(1, 0), t=5)).status is LinkStatus.NO_CANDIDATE


def test_transition_table_gates_by_zone_and_time():
    tr = (Transition(("cam1", "salida"), ("cam2", "entrada"), min_s=2, max_s=30),)
    a = assoc(transitions=tr)
    a.upsert(td("cam1", 1, vec(1, 0), t=0, zone="salida"))
    assert (
        a.associate(td("cam2", 9, vec(1, 0), t=10, zone="entrada")).status
        is LinkStatus.LINKED
    )
    r = a.associate(td("cam2", 9, vec(1, 0), t=50, zone="entrada"))
    assert r.status is LinkStatus.NO_CANDIDATE and r.evidence["rejected"] == {
        "transit_time": 1
    }
    assert (
        a.associate(td("cam2", 9, vec(1, 0), t=10, zone="otra")).status
        is LinkStatus.NO_CANDIDATE
    )


def test_result_payload_is_metadata_only():
    a = assoc()
    a.upsert(td("cam1", 1, vec(1, 0)))
    env = a.associate(td("cam2", 9, vec(1, 0), t=5)).to_envelope()
    payload = json.loads(json.dumps(env.payload))
    assert payload["status"] == "linked" and payload["candidate_track"] == 1
    assert "embedding" not in json.dumps(payload)


# --- Market y métricas -------------------------------------------------------------


def test_parse_market_names():
    c = parse(Path("0002_c3s1_000451_03.jpg"))
    assert c is not None and (c.pid, c.cam, c.seq, c.frame) == (2, 3, 1, 451)
    assert parse(Path("-1_c1s1_000001_00.jpg")).pid == -1
    assert parse(Path("Thumbs.db")) is None


def _fake_market(root: Path) -> None:
    import cv2

    for d in ("bounding_box_train", "bounding_box_test", "query"):
        (root / d).mkdir(parents=True)
    img = np.zeros((128, 64, 3), dtype="uint8")
    for pid in range(1, 11):
        for cam in (1, 2):
            cv2.imwrite(
                str(root / "bounding_box_train" / f"{pid:04d}_c{cam}s1_000001_00.jpg"),
                img,
            )
    cv2.imwrite(str(root / "query" / "0050_c1s1_000001_00.jpg"), img)
    cv2.imwrite(str(root / "bounding_box_test" / "0050_c2s1_000001_00.jpg"), img)


def test_splits_have_disjoint_identities(tmp_path: Path):
    _fake_market(tmp_path)
    sp = build_splits(tmp_path, val_ids=3)
    ids = sp.identities()
    assert len(ids["val"]) == 3 and len(ids["train"]) == 7
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"])
    assert all(
        (c.pid, c.cam) not in {(g.pid, g.cam) for g in sp.val_gallery}
        for c in sp.val_query
    )


def crop(pid: int, cam: int) -> Crop:
    return Crop(Path(f"{pid}_{cam}.jpg"), pid, cam, 1, 1)


def test_retrieval_excludes_same_camera_and_scores_rank1():
    q = [crop(1, 1)]
    g = [crop(1, 1), crop(2, 2), crop(1, 2)]
    qf = np.array([[1.0, 0.0]])
    gf = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    gf = gf / np.linalg.norm(gf, axis=1, keepdims=True)
    r = retrieval(qf, q, gf, g)
    assert r["rank1"] == 0.0 and r["rank5"] == 1.0 and r["map"] == pytest.approx(0.5)


def test_association_counts_false_links_separately_from_missed():
    q = [crop(1, 1), crop(2, 1)]
    g = [crop(1, 2), crop(2, 2), crop(3, 3)]
    e = np.eye(3, dtype="float32")
    qf = np.stack([e[0], e[1]])
    gf = np.stack([e[0], e[1], e[2]])
    r = association(qf, q, gf, g, AssociatorConfig(threshold=0.9, margin=0.0))
    assert r["correct_link_rate"] == 1.0 and r["false_link_rate_absent"] == 0.0
    r = association(qf, q, gf, g, AssociatorConfig(threshold=-1.0, margin=0.0))
    # Con umbral nulo, en el escenario "ausente" siempre se fuerza un enlace falso.
    assert r["false_link_rate_absent"] == 1.0


def test_tracks_group_by_identity_and_camera():
    crops = [crop(1, 1), crop(1, 1), crop(1, 2), crop(0, 1), crop(0, 1)]
    feats = np.eye(5, dtype="float32")
    tracks, truth = tracks_from(feats, crops, t=0.0)
    assert len(tracks) == 4  # (1,c1), (1,c2) y dos distractores separados
    assert sorted(truth.values()) == [0, 0, 1, 1]


def test_color_embedder_ignores_head_region():
    a = np.zeros((1, 256, 128, 3), dtype="uint8")
    b = a.copy()
    b[:, :30] = 255  # solo cambia la cabeza
    emb = ColorEmbedder()
    assert float(emb(a)[0] @ emb(b)[0]) == pytest.approx(1.0)


def test_result_envelope_accepts_numpy_ids_and_times():
    a = assoc()
    a.upsert(
        TrackDescriptor("cam1", np.int64(1), np.float32(0), np.float32(0), vec(1, 0))
    )
    r = a.associate(
        TrackDescriptor("cam2", np.int64(9), np.float32(5), np.float32(5), vec(1, 0))
    )
    env = r.to_envelope()
    json.dumps(env.payload)
    assert type(env.payload["source_track"]) is int
    assert type(env.payload["candidate_track"]) is int
    assert type(env.payload["evidence"]["dt_s"]) is float
