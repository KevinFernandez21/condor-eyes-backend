"""Simuladores integrados: mismas formas de payload que los módulos reales."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from system_support import fast_config

from bus import InMemoryHub, MetadataEnvelope, Topic
from compare import FakeDetector
from compare.detectors import Detector
from fusion import IdentityStatus, identity_from_envelope
from system.simulators import (
    ActuatorSimulator,
    IdentitySimulator,
    LocationSimulator,
    MovingDetector,
    identity_payload,
    location_payload,
    ptz_command_payload,
    ptz_health_payload,
    unknown_location_payload,
)
from system.tracking import SimpleTracker, make_track_fn

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def strict(payload) -> None:
    """El bus real lo rechazaría si no fuese JSON estricto."""
    MetadataEnvelope(source="t", payload=payload)
    json.dumps(payload, allow_nan=False)


# --- payloads ---------------------------------------------------------------


def test_location_payload_sigue_la_forma_de_issue_32():
    p = location_payload("person-sim-01", "vault", 0.9, NOW)
    for key in (
        "tag_ref",
        "person_ref",
        "status",
        "zone_id",
        "confidence",
        "unknown_reason",
        "degraded",
        "authorized",
        "battery_pct",
        "battery_low",
        "first_evidence_at",
        "last_evidence_at",
        "computed_at",
        "evidence",
    ):
        assert key in p
    assert p["status"] == "located"
    assert p["kind"] == "location.estimate"
    strict(p)


def test_location_unknown_no_inventa_zona():
    p = unknown_location_payload("tag_missing", NOW)
    assert p["status"] == "unknown"
    assert p["zone_id"] is None
    assert p["unknown_reason"] == "tag_missing"
    assert p["degraded"] is True
    strict(p)


def test_identity_payload_es_consumible_por_la_fusion():
    p = identity_payload("cam-01/1", "person-sim-01", 0.91)
    env = MetadataEnvelope(source="identity", payload=p)
    ev = identity_from_envelope("e1", "cam-01/1", env)
    assert ev.status is IdentityStatus.MATCH
    assert ev.person_id == "person-sim-01"
    assert p["requires_operator"] is True
    assert "embedding" not in p
    strict(p)


def test_ptz_payloads():
    c = ptz_command_payload("cam-01", seq=3, pan=10.0, tilt=-5.0, speed=20.0)
    assert c["kind"] == "ptz.command" and c["simulated"] is True
    h = ptz_health_payload("cam-01", acked=3)
    assert h["kind"] == "ptz.health" and h["simulated"] is True
    strict(c)
    strict(h)


# --- detector y tracker -----------------------------------------------------


def test_detectores_cumplen_el_protocolo():
    for det in (MovingDetector(), FakeDetector()):
        assert isinstance(det.name, str)
        det.warmup()
        out = det.infer(None)
        assert out and {"xyxy", "cls", "conf"} <= set(out[0])
        det.close()
    _: Detector = MovingDetector()


def test_moving_detector_se_mueve_y_es_json_finito():
    det = MovingDetector(width=640, height=480)
    a, b = det.infer(None), det.infer(None)
    assert a[0]["xyxy"] != b[0]["xyxy"]
    strict({"d": a + b})


def test_tracker_mantiene_id_estable_y_crea_otro_para_nuevo():
    tr = SimpleTracker()
    t1 = tr.update("cam", [{"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.9}])
    t2 = tr.update("cam", [{"xyxy": [1, 0, 11, 10], "cls": 0, "conf": 0.9}])
    t3 = tr.update(
        "cam",
        [
            {"xyxy": [2, 0, 12, 10], "cls": 0, "conf": 0.9},
            {"xyxy": [100, 100, 120, 130], "cls": 0, "conf": 0.8},
        ],
    )
    assert t1[0]["track_ref"] == t2[0]["track_ref"] == t3[0]["track_ref"]
    assert len({t["track_ref"] for t in t3}) == 2
    assert t1[0]["track_ref"].startswith("cam/")


def test_tracker_sigue_cualquier_clase_y_no_mezcla_clases():
    tr = SimpleTracker()
    first = tr.update(
        "cam",
        [
            {"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.9},
            {"xyxy": [0, 0, 10, 10], "cls": 2, "conf": 0.9, "label": "car"},
        ],
    )
    assert {t["cls"] for t in first} == {0, 2}
    assert len({t["track_ref"] for t in first}) == 2
    again = tr.update(
        "cam",
        [
            {"xyxy": [1, 0, 11, 10], "cls": 2, "conf": 0.9},
            {"xyxy": [1, 0, 11, 10], "cls": 0, "conf": 0.9},
        ],
    )
    assert {t["track_ref"] for t in again} == {t["track_ref"] for t in first}
    assert next(t for t in first if t["cls"] == 2)["label"] == "car"


def test_tracker_ignora_cajas_invalidas_y_olvida_tracks_viejos():
    tr = SimpleTracker(max_age=2)
    first = tr.update("cam", [{"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.9}])
    assert tr.update("cam", [{"xyxy": [0, 0, 10], "cls": 0, "conf": 0.9}]) == []
    tr.update("cam", [])
    tr.update("cam", [])
    again = tr.update("cam", [{"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.9}])
    assert again[0]["track_ref"] != first[0]["track_ref"]


def test_tracker_respeta_track_id_externo():
    tr = SimpleTracker()
    out = tr.update(
        "cam-02",
        [{"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.9, "track_id": 42}],
    )
    assert out[0]["track_ref"] == "cam-02/42" and out[0]["track_id"] == 42


def test_track_fn_asigna_zona_por_posicion_horizontal():
    cfg = fast_config()
    fn = make_track_fn(cfg.zones)
    left = MetadataEnvelope(
        source="pipeline",
        stream_id="cam-01",
        payload={
            "stream_id": "cam-01",
            "frame_index": 0,
            "timestamp": NOW.isoformat(),
            "width": 100,
            "height": 100,
            "detections": [{"xyxy": [0, 0, 10, 10], "cls": 0, "conf": 0.9}],
        },
    )
    right = MetadataEnvelope(
        source="pipeline",
        stream_id="cam-01",
        payload={
            "stream_id": "cam-01",
            "frame_index": 1,
            "timestamp": NOW.isoformat(),
            "width": 100,
            "height": 100,
            "detections": [{"xyxy": [90, 0, 99, 10], "cls": 0, "conf": 0.9}],
        },
    )
    zl = fn(left)["tracks"][0]["zone_id"]
    zr = fn(right)["tracks"][0]["zone_id"]
    assert zl != zr
    assert {zl, zr} <= {z.zone_id for z in cfg.zones}
    strict(fn(left))


# --- componentes asíncronos -------------------------------------------------


async def collect(hub: InMemoryHub, topic: Topic, seconds: float):
    sub = hub.subscribe(topic)
    await asyncio.sleep(seconds)
    sub.close()
    return [m async for m in sub]


async def test_location_simulator_publica_estimaciones_y_se_detiene():
    cfg = fast_config()
    hub = InMemoryHub()
    sim = LocationSimulator(hub, cfg)
    before = {t for t in asyncio.all_tasks()}
    await sim.start()
    msgs = await collect(hub, sim.topic, 0.3)
    await sim.stop()
    assert msgs and all(m.payload["status"] in {"located", "unknown"} for m in msgs)
    assert {t for t in asyncio.all_tasks()} - before == set() or all(
        t.done() for t in asyncio.all_tasks() - before
    )
    assert sim.health()["status"] == "simulated"
    await hub.close()


async def test_location_simulator_en_modo_degradado_publica_unknown():
    cfg = fast_config()
    hub = InMemoryHub()
    sim = LocationSimulator(hub, cfg, degraded_reason="tag_missing")
    await sim.start()
    msgs = await collect(hub, sim.topic, 0.2)
    await sim.stop()
    assert msgs and all(m.payload["status"] == "unknown" for m in msgs)
    assert sim.health()["status"] == "degraded"
    await hub.close()


async def test_identity_simulator_usa_tracks_del_bus():
    cfg = fast_config()
    hub = InMemoryHub()
    sim = IdentitySimulator(hub, cfg)
    await sim.start()
    await hub.publish(
        Topic.TRACKS,
        MetadataEnvelope(
            source="tracker",
            stream_id="cam-01",
            payload={"tracks": [{"track_ref": "cam-01/1", "zone_id": "vault"}]},
        ),
    )
    msgs = await collect(hub, sim.topic, 0.3)
    await sim.stop()
    assert msgs
    assert msgs[0].payload["track_ref"] == "cam-01/1"
    await hub.close()


async def test_actuator_simulator_publica_comandos_y_salud():
    cfg = fast_config()
    hub = InMemoryHub()
    sim = ActuatorSimulator(hub, cfg)
    await sim.start()
    events = asyncio.create_task(collect(hub, Topic.HEALTH, 0.3))
    health = asyncio.create_task(collect(hub, Topic.HEALTH, 0.3))
    await asyncio.sleep(0)
    got_events, got_health = await events, await health
    await sim.stop()
    assert any(m.payload.get("kind") == "ptz.command" for m in got_events)
    assert any(m.payload.get("kind") == "ptz.health" for m in got_health)
    await hub.close()


@pytest.mark.parametrize(
    "cls", [LocationSimulator, IdentitySimulator, ActuatorSimulator]
)
async def test_stop_es_idempotente_y_start_doble_falla(cls):
    hub = InMemoryHub()
    sim = cls(hub, fast_config())
    await sim.stop()  # antes de start: no hace nada
    await sim.start()
    with pytest.raises(RuntimeError):
        await sim.start()
    await sim.stop()
    await sim.stop()
    await hub.close()


# --- seudónimos y modo escena -------------------------------------------------


def _scene(seed: int = 3, **overrides):
    import dataclasses

    from system.scenes import SceneSimulator

    cfg = fast_config("sim")
    scenes = dataclasses.replace(cfg.scenes, **overrides)
    handles = tuple(cfg.permissions)
    return cfg, SceneSimulator(cfg.cameras, scenes, handles=handles, seed=seed)


async def test_ubicacion_legacy_publica_seudonimos_no_handles():
    from system.pseudonym import Pseudonymizer

    ps = Pseudonymizer(b"k")
    sim = LocationSimulator(InMemoryHub(), fast_config("sim"), pseudonymizer=ps)
    payloads = sim._payloads()
    assert payloads
    refs = {p["person_ref"] for p in payloads}
    assert all(r.startswith("p-") and "person-sim" not in r for r in refs)
    assert all(p["tag_ref"].startswith("tag-") for p in payloads)


async def test_ubicacion_en_escena_sigue_a_su_persona_y_omite_sin_tag():
    from system.pseudonym import Pseudonymizer
    from system.scenes import Role

    cfg, scene = _scene(
        seed=4, stranger_ratio=0.0, no_tag_ratio=0.5, spawn_rate_per_s=3.0
    )
    ps = Pseudonymizer(b"k")
    sim = LocationSimulator(InMemoryHub(), cfg, pseudonymizer=ps, scene=scene)
    checked = False
    for _ in range(400):
        for cam in scene.camera_ids:
            scene.step(cam)
        holders = scene.holders()
        payloads = {p["person_ref"]: p for p in sim._payloads()}
        for handle, role in holders.items():
            alias = ps.person(handle)
            if role is Role.STAFF_NO_TAG:
                assert alias not in payloads
            else:
                assert payloads[alias]["zone_id"] == scene.zone_of(handle)
                checked = True
    assert checked
    zones = {z.zone_id for z in cfg.effective_zones()}
    assert {p["zone_id"] for p in payloads.values()} <= zones


async def test_identidad_en_escena_segun_perfil():
    import dataclasses

    from system.pseudonym import Pseudonymizer
    from system.scenes import Role

    cfg, scene = _scene(
        seed=5, spawn_rate_per_s=3.0, stranger_ratio=0.4, no_tag_ratio=0.3
    )
    ps = Pseudonymizer(b"k")
    sim = IdentitySimulator(InMemoryHub(), cfg, pseudonymizer=ps, scene=scene, seed=1)
    statuses = set()
    for _ in range(400):
        for cam in scene.camera_ids:
            for d in scene.step(cam):
                if d["label"] != "person":
                    continue
                ref = f"{cam}/{d['track_id']}"
                payload = sim._result(ref, tuple(cfg.permissions))
                profile = scene.profile(ref)
                assert payload is not None and payload["kind"] == "identity.result"
                if profile.role is Role.STRANGER:
                    assert (
                        payload["status"] == "unknown" and payload["person_id"] is None
                    )
                else:
                    assert payload["status"] in {"match", "no_face"}
                    if payload["status"] == "match":
                        assert payload["person_id"] == ps.person(profile.handle)
                        assert payload["person_id"].startswith("p-")
                statuses.add(payload["status"])
                strict(payload)
    assert {"match", "unknown"} <= statuses
    del dataclasses
