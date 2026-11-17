"""SystemApp sirviendo la API de observabilidad (#42) sobre el tap del bus."""

from __future__ import annotations

import asyncio
import dataclasses
import time

import httpx
import pytest
from system_support import CONFIG_PATH, fast_config, other_tasks, thread_names
from test_system_app import no_plugins, wait_for

import system
from comms import BusTap, TapSystemView
from system import SystemApp
from system.config import ApiConfig, ConfigError, load_system_config


def api_config(**api_overrides):
    cfg = fast_config("sim")
    api = dataclasses.replace(cfg.api, enabled=True, port=0, **api_overrides)
    return dataclasses.replace(cfg, api=api)


async def get(url: str, path: str, **kwargs):
    async with httpx.AsyncClient(timeout=5.0) as client:
        return await client.get(url + path, **kwargs)


# --- config -----------------------------------------------------------------


def test_config_trae_api_apagada_y_origenes_del_dashboard():
    cfg = load_system_config(CONFIG_PATH, "sim")
    assert cfg.api.enabled is False
    assert cfg.api.host == "127.0.0.1"
    assert "http://127.0.0.1:8080" in cfg.api.allowed_origins
    assert "http://localhost:8080" in cfg.api.allowed_origins


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("port = 70000", "máximo"),
        ("port = -1", "puerto"),
        ('allowed_origins = "http://x"', "lista"),
        ("allowed_origins = [1]", "texto"),
        ("bogus = 1", "desconocidas"),
        ("host = 3", "texto"),
    ],
)
def test_api_config_invalida(tmp_path, body, match):
    toml = (
        '[system]\n[[zones]]\nzone_id="a"\nx_min=0.0\nx_max=1.0\n'
        f"[api]\n{body}\n[profiles.sim.camera]\nkind='fake'\n"
    )
    path = tmp_path / "s.toml"
    path.write_text(toml, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_system_config(path, "sim")


def test_api_dataclass_valores_por_defecto():
    assert ApiConfig().enabled is False


# --- unificación de estadísticas ---------------------------------------------


def test_instrumented_hub_ya_no_existe():
    assert not hasattr(system, "InstrumentedHub")


async def test_el_tap_es_la_unica_fuente_de_estadisticas():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.start()
    try:
        assert isinstance(app.view, TapSystemView)
        assert isinstance(app.tap, BusTap)
        assert not hasattr(app.hub, "topic_stats")
        await wait_for(lambda: app.hub.stats["published"] > 20)
        await asyncio.sleep(0.2)
        assert len(app.view.topics()) == 7
    finally:
        await app.stop()


# --- API --------------------------------------------------------------------


async def test_api_apagada_por_defecto():
    app = SystemApp(fast_config("sim"), plugins=no_plugins())
    await app.start()
    try:
        assert app.api_url is None
        assert app.snapshot()["api"] is None
    finally:
        await app.stop()


async def test_api_sirve_health_agents_topics_events_decisions():
    app = SystemApp(api_config(), plugins=no_plugins())
    await app.start()
    try:
        url = app.api_url
        assert url and url.startswith("http://127.0.0.1:")
        assert app.snapshot()["api"] == url
        assert await wait_for(
            lambda: any(
                t["topic"] == "events" and t["count"] > 0 for t in app.view.topics()
            )
        )
        health = (await get(url, "/health")).json()
        assert health["status"] == "ok"
        assert health["agents_running"] >= 7
        agents = (await get(url, "/agents")).json()["agents"]
        roles = {a["role"] for a in agents}
        assert {"ingest", "inference", "tracker", "event", "storage", "comms"} <= roles
        assert "supervisor" in roles
        components = {a["name"]: a for a in agents if a["role"] == "component"}
        assert {"camera", "detector", "fusion"} <= {n.split("/")[1] for n in components}
        topics = (await get(url, "/topics")).json()["topics"]
        assert {t["topic"] for t in topics} >= {"vision.detections", "events"}
        assert next(t for t in topics if t["topic"] == "vision.detections")["count"] > 0
        events = (await get(url, "/events?limit=5")).json()["events"]
        assert events
        assert await wait_for(lambda: bool(app.tap.decisions(1)))
        decisions = (await get(url, "/decisions")).json()["decisions"]
        assert decisions and "decision_id" in decisions[0]["payload"]
    finally:
        await app.stop()


async def test_api_con_token_exige_credencial():
    app = SystemApp(api_config(), plugins=no_plugins(), api_token="s3cret")
    await app.start()
    try:
        url = app.api_url
        assert (await get(url, "/health")).status_code == 401
        ok = await get(url, "/health", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
    finally:
        await app.stop()


async def test_api_host_no_local_sin_token_se_niega():
    cfg = api_config(host="0.0.0.0")
    app = SystemApp(cfg, plugins=no_plugins())
    with pytest.raises(Exception, match="token"):
        await app.start()
    assert app.snapshot()["running"] is False


async def test_api_deja_de_responder_y_no_filtra_tareas_ni_hilos():
    threads = thread_names()
    tasks = other_tasks()
    app = SystemApp(api_config(), plugins=no_plugins())
    await app.start()
    url = app.api_url
    assert (await get(url, "/health")).status_code == 200
    await app.stop()
    await asyncio.sleep(0.1)
    with pytest.raises(httpx.TransportError):
        await get(url, "/health")
    assert other_tasks() - tasks == set()
    assert {n for n in thread_names() - threads if n.startswith("video")} == set()


async def test_api_no_expone_frames_ni_embeddings():
    app = SystemApp(api_config(), plugins=no_plugins())
    await app.start()
    try:
        await wait_for(lambda: bool(app.tap.events(1)))
        started = time.monotonic()
        body = (await get(app.api_url, "/events?limit=200")).text.lower()
        assert time.monotonic() - started < 5
        for forbidden in ('"embedding"', '"frame"', '"image"', '"pixels"'):
            assert forbidden not in body
    finally:
        await app.stop()


async def test_puerto_ocupado_falla_con_mensaje_claro_y_apaga_limpio():
    import socket

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    try:
        cfg = api_config()
        cfg = dataclasses.replace(cfg, api=dataclasses.replace(cfg.api, port=port))
        app = SystemApp(cfg, plugins=no_plugins())
        tasks = other_tasks()
        with pytest.raises(RuntimeError, match=f"puerto {port}"):
            await app.start()
        assert app.snapshot()["running"] is False
        await asyncio.sleep(0.05)
        assert other_tasks() - tasks == set()
    finally:
        blocker.close()


async def test_health_cuenta_solo_los_siete_roles_y_separa_componentes():
    app = SystemApp(api_config(), plugins=no_plugins())
    await app.start()
    try:
        await wait_for(lambda: len(app.view.agents()) > 7)
        health = (await get(app.api_url, "/health")).json()
        assert health["agents_total"] == 7
        assert health["agents_running"] == 7
        assert health["agents_failed"] == 0
        assert health["components_total"] >= 6
        assert health["components_ok"] >= 1
        assert health["components_total"] >= health["components_ok"]
        assert health["status"] == "ok"
    finally:
        await app.stop()
