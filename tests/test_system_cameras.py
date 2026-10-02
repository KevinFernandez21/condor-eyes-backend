"""Metadata por cámara (STREAM_STATUS) y director de caídas simuladas."""

from __future__ import annotations

import asyncio
import json

from system_support import fast_config

from bus import InMemoryHub, MetadataEnvelope, Topic
from system.cameras import CameraStatusPublisher, ScenarioDirector, camera_state
from system.sources import fake_source_factory


def test_camera_state_mapeo():
    assert camera_state(None)[0] == "offline"
    ok = {"state": "connected", "retry_count": 0, "last_error": None}
    assert camera_state(ok)[0] == "ok"
    assert (
        camera_state({"state": "connecting", "retry_count": 0, "last_error": None})[0]
        == "degraded"
    )
    assert (
        camera_state({"state": "reconnecting", "retry_count": 1, "last_error": "x"})[0]
        == "degraded"
    )
    assert (
        camera_state({"state": "reconnecting", "retry_count": 3, "last_error": "x"})[0]
        == "offline"
    )
    assert (
        camera_state({"state": "failed", "retry_count": 9, "last_error": "x"})[0]
        == "offline"
    )
    assert (
        camera_state({"state": "stopped", "retry_count": 0, "last_error": None})[0]
        == "offline"
    )


async def collect(hub, seconds):
    sub = hub.subscribe(Topic.STREAM_STATUS)
    await asyncio.sleep(seconds)
    sub.close()
    return [m async for m in sub]


async def test_publisher_emite_la_forma_documentada_por_camara():
    cfg = fast_config("sim")
    hub = InMemoryHub()
    streams = {
        c.camera_id: {
            "state": "connected",
            "retry_count": 0,
            "last_error": None,
            "frames_received": 10,
            "last_frame_at": None,
        }
        for c in cfg.cameras
    }
    pub = CameraStatusPublisher(hub, cfg, lambda: streams)
    await pub.start()
    msgs = await collect(hub, 0.4)
    await pub.stop()
    await hub.close()
    payloads = [m.payload for m in msgs if m.payload.get("kind") == "camera.status"]
    assert {p["camera_id"] for p in payloads} == {
        "cam-01",
        "cam-02",
        "cam-03",
        "cam-04",
    }
    p = next(p for p in payloads if p["camera_id"] == "cam-01")
    assert set(p) == {
        "kind",
        "camera_id",
        "stream_id",
        "name",
        "scene",
        "zones",
        "fps",
        "measured_fps",
        "state",
        "stream_state",
        "resolution",
        "frames_received",
        "last_frame_at",
        "last_error",
        "simulated",
    }
    assert p["name"] == cfg.cameras[0].name
    assert p["state"] == "ok"
    assert p["resolution"] == {"width": 1920, "height": 1080}
    assert {"zone_id", "restricted"} <= set(p["zones"][0])
    assert all(
        m.stream_id == m.payload["camera_id"]
        for m in msgs
        if m.payload.get("kind") == "camera.status"
    )
    json.dumps(p, allow_nan=False)
    MetadataEnvelope(source="t", payload=p)


async def test_publisher_refleja_degradacion_y_recuperacion():
    cfg = fast_config("sim")
    hub = InMemoryHub()
    streams = {
        c.camera_id: {"state": "connected", "retry_count": 0, "last_error": None}
        for c in cfg.cameras
    }
    pub = CameraStatusPublisher(hub, cfg, lambda: streams)
    await pub.start()
    streams["cam-02"] = {
        "state": "reconnecting",
        "retry_count": 1,
        "last_error": "caída",
    }
    await asyncio.sleep(0.25)
    streams["cam-02"] = {
        "state": "reconnecting",
        "retry_count": 4,
        "last_error": "caída",
    }
    await asyncio.sleep(0.25)
    streams["cam-02"] = {"state": "connected", "retry_count": 0, "last_error": None}
    msgs = await collect(hub, 0.3)
    history = [
        m.payload["state"]
        for t, m in hub.history
        if t is Topic.STREAM_STATUS and m.payload.get("camera_id") == "cam-02"
    ]
    await pub.stop()
    await hub.close()
    assert msgs
    assert "degraded" in history and "offline" in history and history[-1] == "ok"


async def test_director_desconecta_y_reconecta_una_camara():
    cfg = fast_config("sim")
    import dataclasses

    scenes = dataclasses.replace(
        cfg.scenes, dropouts=True, dropout_interval_s=0.2, dropout_duration_s=0.3
    )
    cfg = dataclasses.replace(cfg, scenes=scenes)
    factory = fake_source_factory(cfg.cameras)
    hub = InMemoryHub()
    director = ScenarioDirector(hub, cfg, factory, seed=1)
    await director.start()
    seen_down = False
    for _ in range(60):
        if any(not cam._connected for cam in factory.cameras()):
            seen_down = True
        await asyncio.sleep(0.02)
    await director.stop()
    await hub.close()
    assert seen_down
    assert all(cam._connected for cam in factory.cameras()), (
        "debe dejar todo reconectado"
    )
    assert director.health()["dropouts"] >= 1


async def test_director_sin_caidas_no_toca_las_camaras():
    cfg = fast_config("sim")
    import dataclasses

    cfg = dataclasses.replace(
        cfg, scenes=dataclasses.replace(cfg.scenes, dropouts=False)
    )
    factory = fake_source_factory(cfg.cameras)
    hub = InMemoryHub()
    director = ScenarioDirector(hub, cfg, factory, seed=1)
    await director.start()
    await asyncio.sleep(0.3)
    await director.stop()
    await hub.close()
    assert all(cam._connected for cam in factory.cameras())
