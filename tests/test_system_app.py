"""SystemApp: ensamblado real de runtime + pipeline + fusión + simuladores."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time

import pytest
from system_support import fast_config, other_tasks, thread_names

from agents import AgentRuntime
from bus import InMemoryHub, Topic
from compare import FakeDetector
from pipeline import StreamSource
from system import PluginRegistry, SystemApp
from system.config import CameraConfig, TagConfig

ROLES = {"ingest", "inference", "tracker", "event", "storage", "supervisor", "comms"}
FORBIDDEN_KEYS = {"frame", "image", "embedding", "embeddings", "pixels", "data"}


async def wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.02)
    return condition()


def topic_count(app: SystemApp, topic: Topic) -> int:
    return app.snapshot()["topics"][topic.value]["published"]


def no_plugins() -> PluginRegistry:
    return PluginRegistry(finder=lambda module: False)


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk_keys(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from walk_keys(item)


# --- perfil sim -------------------------------------------------------------


async def test_sim_corre_los_siete_roles_y_publica_en_todos_los_topics():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.start()
    try:
        assert isinstance(app.runtime, AgentRuntime)
        assert isinstance(app.hub, InMemoryHub)
        assert await wait_for(
            lambda: all(
                topic_count(app, t) > 0
                for t in (
                    Topic.STREAM_STATUS,
                    Topic.DETECTIONS,
                    Topic.TRACKS,
                    Topic.EVENTS,
                    Topic.HEALTH,
                )
            )
        ), app.snapshot()["topics"]
        snap = app.snapshot()
        assert ROLES <= {name.split("/")[0] for name in snap["agents"]}
        assert all(a["state"] == "running" for a in snap["agents"].values())
        assert snap["profile"] == "sim" and snap["running"] is True
    finally:
        await app.stop()


async def test_sim_fusion_publica_decisiones_en_events():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.start()
    try:

        def has_decision() -> bool:
            return any(
                t is Topic.EVENTS and "outcome" in m.payload for t, m in app.hub.history
            )

        assert await wait_for(has_decision), "la fusión no emitió decisiones"
    finally:
        await app.stop()


async def test_sim_ningun_frame_ni_embedding_en_el_bus():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.start()
    await wait_for(lambda: topic_count(app, Topic.TRACKS) > 3)
    await app.stop()
    assert app.hub.history
    for topic, message in app.hub.history:
        assert not (set(walk_keys(message.payload)) & FORBIDDEN_KEYS), (topic, message)
        json.dumps(message.payload, allow_nan=False)


async def test_snapshot_es_json_estricto_y_estable():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.start()
    await wait_for(lambda: topic_count(app, Topic.DETECTIONS) > 0)
    snap = app.snapshot()
    await app.stop()
    json.dumps(snap, allow_nan=False)
    assert {
        "profile",
        "running",
        "uptime_s",
        "agents",
        "topics",
        "hub",
        "components",
    } <= set(snap)
    assert set(snap["topics"]) == {t.value for t in Topic}
    assert {"published", "last_at"} <= set(snap["topics"]["vision.detections"])
    assert {"published", "dropped", "overflow"} <= set(snap["hub"])
    for name in ("camera", "detector", "location", "identity", "actuation", "fusion"):
        assert name in snap["components"], name
        assert "status" in snap["components"][name]


async def test_plugins_ausentes_se_reportan_not_installed_y_sim_sigue_andando():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.start()
    try:
        components = app.snapshot()["components"]
        for name in ("location", "identity", "actuation", "tagbridge"):
            assert components[name]["plugin"] == "not_installed"
        assert components["location"]["status"] == "simulated"
        assert await wait_for(lambda: topic_count(app, Topic.TRACKS) > 0)
    finally:
        await app.stop()


async def test_salud_de_componentes_llega_a_topic_health():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.start()
    try:
        await wait_for(
            lambda: any(
                t is Topic.HEALTH and m.payload.get("kind") == "component.health"
                for t, m in app.hub.history
            )
        )
        comp = [
            m
            for t, m in app.hub.history
            if t is Topic.HEALTH and m.payload.get("kind") == "component.health"
        ]
        assert comp
        assert {m.payload["component"] for m in comp} >= {"camera", "location"}
        # no debe confundir al supervisor: sin clave "state" que dispare reinicios
        assert all("state" not in m.payload for m in comp)
    finally:
        await app.stop()


# --- ciclo de vida ----------------------------------------------------------


async def test_stop_no_deja_hilos_ni_tareas():
    threads_before = thread_names()
    tasks_before = other_tasks()
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.start()
    await wait_for(lambda: topic_count(app, Topic.TRACKS) > 0)
    await app.stop()
    await asyncio.sleep(0.05)
    assert other_tasks() - tasks_before == set()
    leaked = {n for n in thread_names() - threads_before if n.startswith("video")}
    assert leaked == set()
    assert app.snapshot()["running"] is False


async def test_stop_idempotente_y_start_doble_falla():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.stop()  # antes de start: no hace nada
    await app.start()
    with pytest.raises(RuntimeError, match="ya está en ejecución"):
        await app.start()
    await app.stop()
    await app.stop()


async def test_run_con_duracion_vuelve_sola():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    started = time.monotonic()
    await app.run(duration=0.4)
    assert 0.3 <= time.monotonic() - started < 3.0
    assert app.snapshot()["running"] is False


async def test_run_se_detiene_con_el_evento_externo():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    stop = asyncio.Event()
    task = asyncio.create_task(app.run(stop_event=stop))
    await wait_for(lambda: app.snapshot()["running"])
    stop.set()
    await asyncio.wait_for(task, 5.0)
    assert app.snapshot()["running"] is False


async def test_cancelar_run_tambien_apaga_limpio():
    tasks_before = other_tasks()
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    task = asyncio.create_task(app.run())
    await wait_for(lambda: app.snapshot()["running"])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert app.snapshot()["running"] is False
    await asyncio.sleep(0.05)
    assert other_tasks() - tasks_before == set()


# --- degradación ------------------------------------------------------------


class DeadCamera:
    """Fuente que nunca abre: simula una webcam ausente."""

    def __init__(self, _source: StreamSource) -> None:
        pass

    def open(self) -> None:
        raise ConnectionError("no hay webcam")

    def read(self):  # pragma: no cover
        raise ConnectionError("no hay webcam")

    def close(self) -> None:
        pass


async def test_laptop_sin_webcam_ni_c6_degrada_visible_y_no_cae():
    app = SystemApp(
        fast_config("laptop"),
        plugins=no_plugins(),
        source_factory=DeadCamera,
        detector=FakeDetector(latency_ms=0),
    )
    await app.start()
    try:

        def degraded() -> bool:
            c = app.snapshot()["components"]
            return c["camera"]["status"] == "degraded"

        assert await wait_for(degraded)
        comps = app.snapshot()["components"]
        assert comps["tagbridge"]["plugin"] == "not_installed"
        assert comps["location"]["status"] == "degraded"
        # location "unknown" visible en el bus mientras no hay C6
        assert await wait_for(
            lambda: any(
                m.payload.get("kind") == "location.estimate"
                and m.payload["status"] == "unknown"
                for _, m in app.hub.history
            )
        )
        assert all(a["state"] == "running" for a in app.snapshot()["agents"].values())
    finally:
        await app.stop()


async def test_detector_que_falla_degrada_pero_el_pipeline_sigue():
    class BrokenDetector:
        name = "roto"

        def warmup(self, n: int = 5) -> None:
            raise RuntimeError("sin pesos")

        def infer(self, frame):
            raise RuntimeError("sin pesos")

        def close(self) -> None:
            pass

    app = SystemApp(fast_config("sim"), plugins=no_plugins(), detector=BrokenDetector())
    await app.start()
    try:
        assert await wait_for(
            lambda: app.snapshot()["components"]["detector"]["status"] == "degraded"
        )
        assert await wait_for(lambda: topic_count(app, Topic.DETECTIONS) > 0)
        assert "sin pesos" in app.snapshot()["components"]["detector"]["detail"]
    finally:
        await app.stop()


async def test_detector_yolo_sin_ultralytics_ni_pesos_degrada(monkeypatch, tmp_path):
    cfg = fast_config("sim")
    cfg = dataclasses.replace(
        cfg,
        detector=dataclasses.replace(
            cfg.detector, kind="yolov8n", weights=str(tmp_path / "no-existe.pt")
        ),
    )
    app = SystemApp(cfg, plugins=no_plugins())
    await app.start()
    try:
        assert await wait_for(
            lambda: app.snapshot()["components"]["detector"]["status"] == "degraded"
        )
    finally:
        await app.stop()


# --- replay -----------------------------------------------------------------


def make_video(path, frames: int = 8) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 10, dtype="uint8"))
    writer.release()


async def test_replay_con_video_y_jsonl(tmp_path):
    video = tmp_path / "clip.avi"
    make_video(video)
    tags = tmp_path / "tags.jsonl"
    tags.write_text(
        "\n".join(
            json.dumps(line)
            for line in (
                {
                    "offset_s": 0.0,
                    "person_ref": "person-sim-01",
                    "zone_id": "vault",
                    "confidence": 0.9,
                },
                {
                    "offset_s": 0.05,
                    "person_ref": "person-sim-02",
                    "zone_id": "lobby",
                    "confidence": 0.8,
                },
            )
        ),
        encoding="utf-8",
    )
    cfg = fast_config("replay")
    cfg = dataclasses.replace(
        cfg,
        camera=CameraConfig(
            kind="file", stream_id="replay-01", path=str(video), fps=50.0
        ),
        tag=TagConfig(kind="replay", path=str(tags), interval_s=0.05),
    )
    app = SystemApp(cfg, plugins=no_plugins(), detector=FakeDetector(latency_ms=0))
    await app.start()
    try:
        assert await wait_for(lambda: topic_count(app, Topic.DETECTIONS) > 2)
        assert await wait_for(
            lambda: (
                {
                    m.payload.get("person_ref")
                    for _, m in app.hub.history
                    if m.payload.get("kind") == "location.estimate"
                }
                >= {"person-sim-01", "person-sim-02"}
            )
        )
        assert app.snapshot()["components"]["camera"]["status"] == "ok"
    finally:
        await app.stop()


async def test_replay_sin_jsonl_degrada_a_unknown(tmp_path):
    video = tmp_path / "clip.avi"
    make_video(video)
    cfg = fast_config("replay")
    cfg = dataclasses.replace(
        cfg,
        camera=CameraConfig(
            kind="file", stream_id="replay-01", path=str(video), fps=50.0
        ),
        tag=TagConfig(
            kind="replay", path=str(tmp_path / "falta.jsonl"), interval_s=0.05
        ),
    )
    app = SystemApp(cfg, plugins=no_plugins(), detector=FakeDetector(latency_ms=0))
    await app.start()
    try:
        assert await wait_for(
            lambda: app.snapshot()["components"]["location"]["status"] == "degraded"
        )
    finally:
        await app.stop()


async def test_replay_con_video_inexistente_degrada_sin_caerse(tmp_path):
    cfg = fast_config("replay")
    cfg = dataclasses.replace(
        cfg,
        camera=CameraConfig(
            kind="file", stream_id="replay-01", path=str(tmp_path / "no.avi"), fps=50.0
        ),
        tag=TagConfig(kind="none"),
    )
    app = SystemApp(cfg, plugins=no_plugins(), detector=FakeDetector(latency_ms=0))
    await app.start()
    try:
        assert await wait_for(
            lambda: app.snapshot()["components"]["camera"]["status"] == "degraded"
        )
    finally:
        await app.stop()


async def test_pesos_faltantes_explican_como_obtenerlos(tmp_path, monkeypatch):
    monkeypatch.delenv("CONDOR_WEIGHTS", raising=False)
    cfg = fast_config("sim")
    missing = str(tmp_path / "no-existe.pt")
    cfg = dataclasses.replace(
        cfg, detector=dataclasses.replace(cfg.detector, kind="yolov8n", weights=missing)
    )
    app = SystemApp(cfg, plugins=no_plugins())
    await app.start()
    try:
        assert await wait_for(
            lambda: app.snapshot()["components"]["detector"]["status"] == "degraded"
        )
        detail = app.snapshot()["components"]["detector"]["detail"]
        assert missing in detail
        assert "CONDOR_WEIGHTS" in detail
        assert "ultralytics" in detail
        assert "no se descarga" in detail
    finally:
        await app.stop()


async def test_env_condor_weights_tiene_prioridad(tmp_path, monkeypatch):
    env_path = str(tmp_path / "desde-env.pt")
    monkeypatch.setenv("CONDOR_WEIGHTS", env_path)
    cfg = fast_config("sim")
    cfg = dataclasses.replace(
        cfg,
        detector=dataclasses.replace(
            cfg.detector, kind="yolov8n", weights=str(tmp_path / "config.pt")
        ),
    )
    app = SystemApp(cfg, plugins=no_plugins())
    await app.start()
    try:
        assert await wait_for(
            lambda: app.snapshot()["components"]["detector"]["status"] == "degraded"
        )
        assert env_path in app.snapshot()["components"]["detector"]["detail"]
    finally:
        await app.stop()


async def test_el_detector_no_se_cierra_con_una_inferencia_en_curso():
    import threading

    events: list[str] = []
    entered = threading.Event()

    class SlowDetector:
        name = "lento"

        def warmup(self, n: int = 5) -> None:
            return None

        def infer(self, frame):
            events.append("infer-start")
            entered.set()
            time.sleep(0.8)
            events.append("infer-end")
            return []

        def close(self) -> None:
            events.append("close")

    cfg = dataclasses.replace(fast_config("sim"), shutdown_timeout_s=0.2)
    app = SystemApp(cfg, plugins=no_plugins(), detector=SlowDetector())
    await app.start()
    assert await asyncio.get_running_loop().run_in_executor(None, entered.wait, 3.0)
    await app.stop()
    await asyncio.sleep(1.0)
    assert events.index("close") > events.index("infer-end"), events
    assert events.count("infer-start") == 1, events


async def test_telemetria_simulada_no_ensucia_events():
    """Ubicación, identidad y PTZ no viajan por EVENTS (solo incidentes y decisiones)."""
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.start()
    await wait_for(lambda: topic_count(app, Topic.EVENTS) > 2)
    await asyncio.sleep(0.3)
    await app.stop()
    kinds_events = {
        m.payload.get("kind") for t, m in app.hub.history if t is Topic.EVENTS
    }
    assert not kinds_events & {"location.estimate", "identity.result", "ptz.command"}
    kinds_all = {m.payload.get("kind") for _, m in app.hub.history}
    assert {"location.estimate", "identity.result", "ptz.command"} <= kinds_all
