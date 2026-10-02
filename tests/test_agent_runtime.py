"""Pruebas de integración del runtime multiagente sobre el bus tipado."""

import ast
import asyncio
import threading
import warnings
from pathlib import Path

import pytest

from agents import MULTIAGENT_ROUTE, get_route
from agents.handlers import SupervisorHandler, build_default_handlers
from agents.runtime import (
    AgentRuntime,
    RetryPolicy,
    RouteViolationError,
    RuntimeStartError,
    WorkerState,
)
from bus import MetadataEnvelope, Topic
from bus.agentscope_hub import AgentScopeHub
from bus.memory import InMemoryHub

ROLES = {"ingest", "inference", "tracker", "event", "storage", "supervisor", "comms"}


class CountingSink:
    """Sink NO idempotente: si ve duplicados, los cuenta (prueba la deduplicación)."""

    def __init__(self) -> None:
        self.saved: list[MetadataEnvelope] = []

    async def save(self, envelope: MetadataEnvelope) -> None:
        self.saved.append(envelope)

    async def send(self, envelope: MetadataEnvelope) -> None:
        self.saved.append(envelope)


def person_rule(track: MetadataEnvelope):
    return [
        {
            "type": "person_detected",
            "track_ids": [t["id"] for t in track.payload["tracks"]],
        }
    ]


@pytest.fixture(params=["memory", "agentscope"])
def hub(request):
    return InMemoryHub() if request.param == "memory" else AgentScopeHub()


@pytest.fixture
def sinks():
    return CountingSink(), CountingSink()


@pytest.fixture
def runtime(hub, sinks):
    storage, alerts = sinks
    handlers = build_default_handlers(
        storage_sink=storage, alert_sink=alerts, event_rules=[person_rule]
    )
    return AgentRuntime(hub, handlers, retry=RetryPolicy(max_attempts=3, base_delay=0))


DETECTION = {"detections": [{"id": 1, "label": "person", "conf": 0.9}]}


async def test_all_seven_roles_start_and_stop_cleanly(runtime):
    await runtime.start()
    health = runtime.health()
    assert {h["role"] for h in health.values()} == ROLES
    assert all(h["state"] == WorkerState.RUNNING for h in health.values())

    await runtime.stop()
    assert all(h["state"] == WorkerState.STOPPED for h in runtime.health().values())
    leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert leftovers == []


async def test_start_is_not_reentrant_and_stop_is_idempotent(runtime):
    await runtime.start()
    with pytest.raises(RuntimeError, match="ya está en ejecución"):
        await runtime.start()
    await runtime.stop()
    await runtime.stop()


async def test_metadata_flows_through_the_canonical_route(runtime, sinks):
    storage, alerts = sinks
    await runtime.start()
    detection = await runtime.emit(
        "inference", Topic.DETECTIONS, DETECTION, stream_id="cam-01"
    )
    await runtime.wait_idle()
    await runtime.stop()

    assert len(storage.saved) == 1
    assert len(alerts.saved) == 1
    event = storage.saved[0]
    assert event.source == "event"
    assert event.stream_id == "cam-01"
    assert event.correlation_id == detection.correlation_id
    assert event.event_id == f"{detection.event_id}/tracks/events/0"
    assert event.payload["type"] == "person_detected"


async def test_duplicate_event_ids_do_not_duplicate_side_effects(runtime, sinks):
    storage, alerts = sinks
    await runtime.start()
    detection = MetadataEnvelope(
        source="inference", payload=DETECTION, stream_id="cam-01", event_id="det-1"
    )
    await runtime.emit_envelope("inference", Topic.DETECTIONS, detection)
    await runtime.emit_envelope("inference", Topic.DETECTIONS, detection)
    duplicate_event = MetadataEnvelope(
        source="event", payload={"type": "x"}, event_id="evt-dup"
    )
    await runtime.emit_envelope("event", Topic.EVENTS, duplicate_event)
    await runtime.emit_envelope("event", Topic.EVENTS, duplicate_event)
    await runtime.wait_idle()
    await runtime.stop()

    assert sorted(m.event_id for m in storage.saved) == [
        "det-1/tracks/events/0",
        "evt-dup",
    ]
    assert sorted(m.event_id for m in alerts.saved) == [
        "det-1/tracks/events/0",
        "evt-dup",
    ]
    assert runtime.health()["storage"]["duplicates"] >= 1


async def test_transient_failure_is_retried_without_error_event(hub):
    attempts = 0

    class Flaky:
        async def start(self) -> None: ...
        async def stop(self) -> None: ...
        async def handle(self, topic, envelope):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("SQLite bloqueado")
            return []

    handlers = build_default_handlers(
        storage_sink=CountingSink(), alert_sink=CountingSink()
    )
    handlers["storage"] = Flaky()
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    runtime = AgentRuntime(
        hub,
        handlers,
        retry=RetryPolicy(max_attempts=3, base_delay=0.5, backoff=2.0),
        sleep=fake_sleep,
    )
    errors = hub.subscribe(Topic.ERRORS)
    await runtime.start()
    await runtime.emit("event", Topic.EVENTS, {"type": "x"})
    await runtime.wait_idle()
    await runtime.stop()
    await hub.close()

    assert attempts == 3
    assert sleeps == [0.5, 1.0]
    assert runtime.health()["storage"]["retries"] == 2
    assert [m async for m in errors] == []


async def test_exhausted_retries_publish_error_event_and_allow_redelivery(hub):
    fail = True

    class Broken:
        async def start(self) -> None: ...
        async def stop(self) -> None: ...
        async def handle(self, topic, envelope):
            if fail:
                raise OSError("disco lleno")
            return []

    handlers = build_default_handlers(
        storage_sink=CountingSink(), alert_sink=CountingSink()
    )
    handlers["storage"] = Broken()

    async def no_sleep(delay: float) -> None: ...

    runtime = AgentRuntime(
        hub, handlers, retry=RetryPolicy(max_attempts=2, base_delay=0), sleep=no_sleep
    )
    errors = hub.subscribe(Topic.ERRORS)
    await runtime.start()
    event = MetadataEnvelope(source="event", payload={"type": "x"}, event_id="evt-1")
    await runtime.emit_envelope("event", Topic.EVENTS, event)
    await runtime.wait_idle()

    health = runtime.health()["storage"]
    assert health["failures"] == 1
    assert "disco lleno" in health["last_error"]

    # Un fallo no marca el evento como procesado: la reentrega se reintenta.
    fail = False
    await runtime.emit_envelope("event", Topic.EVENTS, event)
    await runtime.wait_idle()
    assert runtime.health()["storage"]["processed"] == 1

    await runtime.stop()
    await hub.close()
    (error,) = [m async for m in errors]
    assert error.source == "storage"
    assert error.causation_id == "evt-1"
    assert error.payload["error_type"] == "OSError"
    assert error.payload["attempts"] == 2


async def test_handler_cannot_publish_outside_its_route(hub):
    class Rogue:
        async def start(self) -> None: ...
        async def stop(self) -> None: ...
        async def handle(self, topic, envelope):
            return [(Topic.COMMANDS, envelope.derive("storage", {}, suffix="x"))]

    handlers = build_default_handlers(
        storage_sink=CountingSink(), alert_sink=CountingSink()
    )
    handlers["storage"] = Rogue()
    runtime = AgentRuntime(
        hub, handlers, retry=RetryPolicy(max_attempts=3, base_delay=0)
    )
    errors = hub.subscribe(Topic.ERRORS)
    await runtime.start()
    await runtime.emit("event", Topic.EVENTS, {"type": "x"})
    await runtime.wait_idle()
    await runtime.stop()
    await hub.close()

    (error,) = [m async for m in errors]
    assert error.payload["error_type"] == RouteViolationError.__name__
    assert error.payload["attempts"] == 1
    assert get_route("storage").publishes == (Topic.HEALTH,)


async def test_emit_rejects_roles_and_topics_outside_the_route(runtime):
    await runtime.start()
    with pytest.raises(RouteViolationError, match="no puede publicar"):
        await runtime.emit("comms", Topic.DETECTIONS, {})
    with pytest.raises(KeyError, match="desconocido"):
        await runtime.emit("nope", Topic.DETECTIONS, {})
    await runtime.stop()


async def test_supervisor_observes_health_and_errors_without_video(runtime, hub):
    supervisor = runtime.handler("supervisor")
    assert isinstance(supervisor, SupervisorHandler)
    assert set(get_route("supervisor").consumes) == {
        Topic.HEALTH,
        Topic.STREAM_STATUS,
        Topic.ERRORS,
    }

    await runtime.start()
    await runtime.emit(
        "ingest", Topic.STREAM_STATUS, {"state": "up"}, stream_id="cam-01"
    )
    await runtime.publish_health()
    await runtime.wait_idle()

    report = supervisor.report()
    assert {"ingest", "storage", "comms"} <= set(report["health"])
    assert report["streams"] == {"cam-01": "up"}
    assert report["health"]["storage"]["state"] == WorkerState.RUNNING
    await runtime.stop()
    # el hub solo vio metadata serializable: ningún mensaje con binarios
    for _, message in getattr(hub, "history", []):
        assert "frame" not in str(message.payload).lower()


async def test_supervisor_requests_restart_of_failed_role(runtime, hub):
    commands = hub.subscribe(Topic.COMMANDS)
    await runtime.start()
    failed = MetadataEnvelope(
        source="tracker",
        payload={"role": "tracker", "state": WorkerState.FAILED.value},
    )
    await runtime.emit_envelope("tracker", Topic.HEALTH, failed)
    await runtime.wait_idle()
    await runtime.stop()
    await hub.close()

    (command,) = [m async for m in commands]
    assert command.source == "supervisor"
    assert command.payload == {"action": "restart", "target": "tracker"}
    assert command.causation_id == failed.event_id


async def test_start_failure_rolls_back_started_workers(hub):
    stopped: list[str] = []

    class Tracking:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name, self.fail = name, fail

        async def start(self) -> None:
            if self.fail:
                raise OSError("sin acceso a la cámara")

        async def stop(self) -> None:
            stopped.append(self.name)

        async def handle(self, topic, envelope):
            return []

    handlers = {role: Tracking(role) for role in ROLES}
    handlers["ingest"] = Tracking("ingest", fail=True)
    runtime = AgentRuntime(hub, handlers)
    with pytest.raises(RuntimeStartError, match="ingest.*sin acceso a la cámara"):
        await runtime.start()

    assert set(stopped) == ROLES - {"ingest"}
    assert [t for t in asyncio.all_tasks() if t is not asyncio.current_task()] == []


def test_missing_handler_is_reported_in_spanish(hub):
    handlers = build_default_handlers(
        storage_sink=CountingSink(), alert_sink=CountingSink()
    )
    del handlers["tracker"]
    with pytest.raises(ValueError, match="Falta el manejador.*tracker"):
        AgentRuntime(hub, handlers)


async def test_ingest_scales_per_camera_and_filters_commands(hub):
    received: list[tuple[str | None, str]] = []

    def factory(instance: str | None):
        from agents.handlers import IngestHandler

        async def on_command(envelope: MetadataEnvelope) -> None:
            received.append((instance, envelope.payload["action"]))

        return IngestHandler(stream_id=instance, on_command=on_command)

    handlers = build_default_handlers(
        storage_sink=CountingSink(), alert_sink=CountingSink()
    )
    handlers["ingest"] = factory
    runtime = AgentRuntime(hub, handlers, instances={"ingest": ["cam-01", "cam-02"]})
    await runtime.start()
    assert {"ingest/cam-01", "ingest/cam-02"} <= set(runtime.health())
    await runtime.emit(
        "supervisor", Topic.COMMANDS, {"action": "restart_stream"}, stream_id="cam-02"
    )
    await runtime.wait_idle()
    await runtime.stop()

    assert received == [("cam-02", "restart_stream")]


async def test_pipeline_threads_can_publish_metadata_via_threadsafe_bridge(
    runtime, sinks
):
    storage, _ = sinks
    await runtime.start()

    def pipeline_callback() -> None:
        future = runtime.emit_threadsafe(
            "inference", Topic.DETECTIONS, DETECTION, stream_id="cam-09"
        )
        future.result(timeout=5)

    thread = threading.Thread(target=pipeline_callback)
    thread.start()
    while thread.is_alive():
        await asyncio.sleep(0.01)
    thread.join()
    await runtime.wait_idle()
    await runtime.stop()

    assert [m.stream_id for m in storage.saved] == ["cam-09"]


async def test_heartbeat_publishes_health_periodically(hub):
    handlers = build_default_handlers(
        storage_sink=CountingSink(), alert_sink=CountingSink()
    )
    runtime = AgentRuntime(hub, handlers, heartbeat_interval=0.01)
    sub = hub.subscribe(Topic.HEALTH)
    await runtime.start()
    first = await asyncio.wait_for(sub.__anext__(), 2)
    assert first.payload["state"] == WorkerState.RUNNING
    await runtime.stop()
    await hub.close()


def test_route_roles_match_runtime_roles():
    assert {route.name for route in MULTIAGENT_ROUTE} == ROLES


# --- Higiene de AgentScope --------------------------------------------------

ALLOWED_AGENTSCOPE_IMPORTS = {
    "agentscope.agent": {"Agent"},
    "agentscope.message": {"Msg", "TextBlock"},
    "agentscope.model": {"ChatModelBase"},
    "agentscope.tool": {"Toolkit"},
}


def _agentscope_imports() -> list[tuple[Path, str, str]]:
    found = []
    for path in Path("src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and (node.module or "").split(".")[0] == "agentscope"
            ):
                found += [(path, node.module or "", alias.name) for alias in node.names]
            elif isinstance(node, ast.Import):
                found += [
                    (path, alias.name, "*")
                    for alias in node.names
                    if alias.name.startswith("agentscope")
                ]
    return found


def test_only_supported_agentscope_2x_apis_are_imported():
    imports = _agentscope_imports()
    assert imports, "se esperaba al menos un import de agentscope"
    for path, module, name in imports:
        assert name in ALLOWED_AGENTSCOPE_IMPORTS.get(module, set()), (
            f"{path}: import no permitido {module}.{name}"
        )


def test_agentscope_imports_emit_no_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for _, module, name in _agentscope_imports():
            imported = __import__(module, fromlist=[name])
            getattr(imported, name)
