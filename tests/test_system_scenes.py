"""Escenas sintéticas multicámara: determinismo, forma de las detecciones y perfiles."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from system_support import fast_config

from bus import MetadataEnvelope
from system.pseudonym import Pseudonymizer
from system.scenes import Role, SceneSimulator

HANDLES = ("person-sim-01", "person-sim-02", "person-sim-03", "person-sim-04")


def make(seed: int = 7, **scene_overrides) -> SceneSimulator:
    import dataclasses

    cfg = fast_config("sim")
    scenes = dataclasses.replace(cfg.scenes, **scene_overrides)
    return SceneSimulator(cfg.cameras, scenes, handles=HANDLES, seed=seed)


def run(sim: SceneSimulator, frames: int, camera: str = "cam-01") -> list[list[dict]]:
    return [sim.step(camera) for _ in range(frames)]


def strip_ts(frames):
    return [
        [{k: v for k, v in d.items() if k != "frame_ts"} for d in f] for f in frames
    ]


def test_config_define_cuatro_camaras_con_escenas_distintas():
    cfg = fast_config("sim")
    assert [c.camera_id for c in cfg.cameras] == [
        "cam-01",
        "cam-02",
        "cam-03",
        "cam-04",
    ]
    assert len({c.name for c in cfg.cameras}) == 4
    assert all((c.width, c.height) == (1920, 1080) for c in cfg.cameras)
    assert all(c.zones for c in cfg.cameras)
    assert any(z.restricted for c in cfg.cameras for z in c.zones)


def test_misma_semilla_misma_secuencia():
    a, b = make(seed=11), make(seed=11)
    assert strip_ts(run(a, 400)) == strip_ts(run(b, 400))


def test_semillas_distintas_difieren():
    assert strip_ts(run(make(seed=1), 400)) != strip_ts(run(make(seed=2), 400))


def test_sin_semilla_es_aleatoria_y_se_expone():
    import dataclasses

    cfg = fast_config("sim")
    a = SceneSimulator(cfg.cameras, cfg.scenes, handles=HANDLES, seed=None)
    b = SceneSimulator(cfg.cameras, cfg.scenes, handles=HANDLES, seed=None)
    assert isinstance(a.seed, int)
    assert a.seed != b.seed
    del dataclasses


def test_forma_de_cada_deteccion():
    sim = make(seed=3)
    seen = []
    for cam in sim.camera_ids:
        for frame in run(sim, 300, cam):
            seen.extend(frame)
    assert seen
    for det in seen:
        assert set(det) == {
            "stream_id",
            "track_id",
            "cls",
            "label",
            "conf",
            "xyxy",
            "frame_ts",
        }
        assert isinstance(det["track_id"], int)
        assert 0.0 < det["conf"] <= 1.0
        x0, y0, x1, y1 = det["xyxy"]
        assert 0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080
        datetime.fromisoformat(det["frame_ts"])
    MetadataEnvelope(source="t", payload={"detections": seen})
    json.dumps(seen, allow_nan=False)


def test_hay_personas_vehiculos_y_movimiento_sin_clase():
    labels = set()
    for seed in range(5):
        sim = make(seed=seed)
        for cam in sim.camera_ids:
            for frame in run(sim, 400, cam):
                labels.update(d["label"] for d in frame)
    assert "person" in labels
    assert labels & {"car", "truck", "bus"}
    assert "motion" in labels


def test_cajas_realistas_persona_alta_vehiculo_ancho_movimiento_sin_clase():
    sim = make(seed=5)
    for cam in sim.camera_ids:
        for frame in run(sim, 500, cam):
            for d in frame:
                x0, y0, x1, y1 = d["xyxy"]
                w, h = x1 - x0, y1 - y0
                if w < 30 or h < 30:
                    continue  # recortada por el borde del encuadre
                if d["label"] == "person" and x0 > 5 and x1 < 1915:
                    assert h > w
                if d["label"] in ("car", "truck", "bus") and x0 > 5 and x1 < 1915:
                    assert w > h
                if d["label"] == "motion":
                    assert d["cls"] == -1


def test_track_id_estable_mientras_el_actor_vive():
    sim = make(seed=9, spawn_rate_per_s=2.0)
    frames = run(sim, 400)
    ids = {}
    for frame in frames:
        for d in frame:
            ids.setdefault(d["track_id"], []).append(d["label"])
    assert any(len(v) > 5 for v in ids.values())
    assert all(len(set(v)) == 1 for v in ids.values())


def test_perfiles_de_personas():
    sim = make(seed=4, stranger_ratio=0.34, no_tag_ratio=0.33, spawn_rate_per_s=2.0)
    roles = set()
    for cam in sim.camera_ids:
        for frame in run(sim, 300, cam):
            for d in frame:
                if d["label"] != "person":
                    continue
                profile = sim.profile(f"{d['stream_id']}/{d['track_id']}")
                assert profile is not None
                roles.add(profile.role)
                if profile.role is Role.STRANGER:
                    assert profile.handle is None
                else:
                    assert profile.handle in HANDLES
    assert roles == {Role.STAFF, Role.STAFF_NO_TAG, Role.STRANGER}


def test_un_handle_no_esta_en_dos_actores_a_la_vez():
    sim = make(seed=6, spawn_rate_per_s=3.0, max_actors=6)
    for _ in range(300):
        live = [
            sim.profile(f"{c}/{d['track_id']}")
            for c in sim.camera_ids
            for d in sim.step(c)
            if d["label"] == "person"
        ]
        handles = [p.handle for p in live if p and p.handle]
        # un handle pertenece como máximo a un actor vivo (entre todas las cámaras)
        assert len(handles) == len(set(handles))


def test_zona_de_un_handle_vivo():
    sim = make(seed=8, stranger_ratio=0.0, no_tag_ratio=0.0, spawn_rate_per_s=2.0)
    found = False
    for _ in range(300):
        for d in sim.step("cam-01"):
            if d["label"] == "person":
                p = sim.profile(f"cam-01/{d['track_id']}")
                if p and p.handle:
                    assert sim.zone_of(p.handle) in {z.zone_id for z in sim.zones}
                    found = True
    assert found


# --- pseudónimos -------------------------------------------------------------


def test_pseudonimo_formato_y_determinismo():
    a = Pseudonymizer(b"k1")
    assert a.person("person-sim-01").startswith("p-")
    assert len(a.person("person-sim-01")) == 6
    assert a.person("person-sim-01") == Pseudonymizer(b"k1").person("person-sim-01")
    assert a.person("person-sim-01") != Pseudonymizer(b"k2").person("person-sim-01")
    assert a.person("person-sim-01") != a.person("person-sim-02")
    assert "person-sim" not in a.person("person-sim-01")
    assert a.tag("person-sim-01").startswith("tag-")


def test_pseudonimo_desde_semilla_y_aleatorio_por_defecto():
    assert Pseudonymizer.from_seed(5).person("x") == Pseudonymizer.from_seed(5).person(
        "x"
    )
    assert Pseudonymizer.random().person("x") != Pseudonymizer.random().person("x")


def test_pseudonimo_detecta_colisiones():
    ps = Pseudonymizer(b"k")
    with pytest.raises(ValueError, match="colisión"):
        ps.check_unique([f"h{i}" for i in range(5000)])
