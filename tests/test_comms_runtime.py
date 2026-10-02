"""El rol comms integrado al AgentRuntime: arranque, parada y API real."""

import ast
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from agents.handlers import build_default_handlers
from agents.runtime import AgentRuntime
from bus import InMemoryHub, Topic
from comms import ConfigurationError, ObservabilityCommsHandler


class Sink:
    async def save(self, envelope):  # EventSink
        pass

    async def send(self, envelope):  # AlertSink
        pass


def build(**comms_kwargs):
    hub = InMemoryHub()
    comms = ObservabilityCommsHandler(hub, port=0, **comms_kwargs)
    handlers = build_default_handlers(
        storage_sink=Sink(),
        alert_sink=Sink(),
        event_rules=[lambda t: [{"type": "person", "zone_id": "z1"}]],
    )
    handlers["comms"] = comms
    runtime = AgentRuntime(hub, handlers, heartbeat_interval=0.05)
    comms.bind_runtime(runtime)
    return hub, runtime, comms


async def test_runtime_arranca_y_detiene_la_api():
    _, runtime, comms = build()
    await runtime.start()
    port = comms.port
    try:
        base = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient() as client:
            await runtime.emit("inference", Topic.DETECTIONS, {"detections": [{"id": 1}]}, stream_id="cam-1")
            await runtime.wait_idle()
            await runtime.publish_health()
            await runtime.wait_idle()
            agents = (await client.get(f"{base}/agents")).json()["agents"]
            assert {a["role"] for a in agents} == {
                "ingest", "inference", "tracker", "event", "storage", "supervisor", "comms",
            }
            assert all("queue_depth" in a for a in agents)
            assert any(a["last_heartbeat"] for a in agents)
            events = (await client.get(f"{base}/events")).json()["events"]
            assert events and events[0]["payload"]["zone_id"] == "z1"
            topics = {t["topic"]: t for t in (await client.get(f"{base}/topics")).json()["topics"]}
            assert topics["vision.detections"]["count"] == 1
            assert (await client.get(f"{base}/health")).json()["status"] == "ok"
    finally:
        await runtime.stop()
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.TransportError):
            await client.get(f"http://127.0.0.1:{port}/health", timeout=1.0)


async def test_el_tap_no_rompe_la_ruta_de_comms():
    _, runtime, _ = build()
    await runtime.start()
    await runtime.emit("event", Topic.EVENTS, {"type": "x"}, stream_id="cam-1")
    await runtime.wait_idle()
    assert runtime.health()["comms"]["processed"] >= 1
    await runtime.stop()


def test_bind_no_local_sin_token_no_arranca():
    with pytest.raises(ConfigurationError):
        ObservabilityCommsHandler(InMemoryHub(), host="0.0.0.0", port=0)


def test_queue_depths_expuesto_por_el_runtime():
    _, runtime, _ = build()
    depths = runtime.queue_depths()
    assert set(depths) == set(runtime.health())
    assert all(v == 0 for v in depths.values())


def test_importar_comms_no_importa_uvicorn_ni_agentscope():
    code = (
        "import sys; sys.path.insert(0, 'src'); import comms, comms.api;"
        "bad = [m for m in ('uvicorn', 'agentscope') if m in sys.modules];"
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_comms_solo_usa_el_hub_tipado():
    root = Path(__file__).resolve().parent.parent / "src" / "comms"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not name.startswith("agentscope"), f"{path.name} importa {name}"
                assert name != "bus.agentscope_hub", f"{path.name} importa {name}"
