"""Pruebas HTTP/WebSocket de la API de observabilidad."""

import json

import pytest
from starlette.testclient import TestClient

from bus import InMemoryHub, MetadataEnvelope, Topic
from comms.api import ConfigurationError, create_app
from comms.tap import BusTap
from comms.view import TapSystemView


class FakeRuntime:
    def health(self):
        return {
            "tracker": {
                "role": "tracker", "instance": None, "state": "running", "processed": 3,
                "duplicates": 0, "failures": 0, "retries": 0, "last_error": None,
            },
            "comms": {
                "role": "comms", "instance": None, "state": "failed", "processed": 1,
                "duplicates": 0, "failures": 2, "retries": 1, "last_error": "X: y",
            },
        }

    def queue_depths(self):
        return {"tracker": 4, "comms": 0}


def make_view():
    tap = BusTap(InMemoryHub())
    return tap, TapSystemView(tap, runtime=FakeRuntime())


def e(payload, stream_id=None, source="s", event_id=None):
    kw = {"event_id": event_id} if event_id else {}
    return MetadataEnvelope(source=source, payload=payload, stream_id=stream_id, **kw)


@pytest.fixture
def client_tap():
    tap, view = make_view()
    tap.record(Topic.HEALTH, e({"role": "tracker", "state": "running"}, source="tracker"))
    tap.record(Topic.COMMANDS, e({"action": "restart", "target": "tracker"}, source="supervisor"))
    tap.record(Topic.EVENTS, e({"type": "loitering", "zone_id": "z1"}, "cam-1"))
    tap.record(
        Topic.EVENTS,
        e({"decision_id": "d1", "outcome": "corroborated", "zone_id": "z1",
           "person_id": "p-7f3a", "tag_id": "tag-ab12"}, "cam-1"),
    )
    tap.record(Topic.EVENTS, e({"decision_id": "d2", "zone_id": "z2"}, "cam-2"))
    return TestClient(create_app(view)), tap


def test_health(client_tap):
    client, _ = client_tap
    body = client.get("/health").json()
    assert body["status"] == "degraded"  # comms está en failed
    assert body["agents_total"] == 2
    assert body["agents_failed"] == 1
    assert "ws_clients" in body and "ws_dropped_total" in body


def test_agents_incluye_heartbeat_reinicios_y_cola(client_tap):
    client, _ = client_tap
    agents = {a["name"]: a for a in client.get("/agents").json()["agents"]}
    tracker = agents["tracker"]
    assert tracker["role"] == "tracker"
    assert tracker["state"] == "running"
    assert tracker["restarts"] == 1
    assert tracker["queue_depth"] == 4
    assert tracker["last_heartbeat"] is not None
    assert agents["comms"]["last_heartbeat"] is None
    assert agents["comms"]["last_error"] == "X: y"


def test_topics(client_tap):
    client, _ = client_tap
    topics = {t["topic"]: t for t in client.get("/topics").json()["topics"]}
    assert topics["events"]["count"] == 3
    assert topics["events"]["last_message_at"] is not None
    assert {"rate_per_s", "drops", "window_s"} <= set(topics["events"])


def test_events_y_decisions_con_filtros_y_limite(client_tap):
    client, _ = client_tap
    assert len(client.get("/events").json()["events"]) == 3
    assert len(client.get("/events?limit=1").json()["events"]) == 1
    dec = client.get("/decisions").json()["decisions"]
    assert [d["payload"]["decision_id"] for d in dec] == ["d2", "d1"]
    z1 = client.get("/decisions?zone=z1").json()["decisions"]
    assert [d["payload"]["decision_id"] for d in z1] == ["d1"]
    cam2 = client.get("/decisions?stream_id=cam-2").json()["decisions"]
    assert [d["payload"]["decision_id"] for d in cam2] == ["d2"]


def test_limite_invalido_es_422(client_tap):
    client, _ = client_tap
    assert client.get("/events?limit=0").status_code == 422
    assert client.get("/events?limit=100000").status_code == 422


def test_identificadores_siguen_seudonimizados_tal_como_en_el_bus(client_tap):
    client, _ = client_tap
    payload = client.get("/decisions?zone=z1").json()["decisions"][0]["payload"]
    assert payload["person_id"] == "p-7f3a"
    assert payload["tag_id"] == "tag-ab12"


def test_respuestas_son_json_finito(client_tap):
    client, _ = client_tap

    def reject(constant):
        raise AssertionError(f"constante no JSON: {constant}")

    for path in ("/health", "/agents", "/topics", "/events", "/decisions"):
        json.loads(client.get(path).text, parse_constant=reject)


def test_no_hay_endpoint_de_video(client_tap):
    client, _ = client_tap
    for path in ("/video", "/preview", "/snapshot", "/stream.mjpeg"):
        assert client.get(path).status_code == 404


def test_solo_lectura(client_tap):
    client, _ = client_tap
    for path in ("/agents", "/topics", "/events", "/decisions", "/health"):
        assert client.post(path).status_code == 405
        assert client.delete(path).status_code == 405


# --- WebSocket ---


def test_ws_recibe_hello_y_envelopes_filtrados_por_topico():
    tap, view = make_view()
    client = TestClient(create_app(view))
    with client.websocket_connect("/ws?topics=events") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["topics"] == ["events"]
        tap.record(Topic.TRACKS, e({"ignored": True}))
        tap.record(Topic.EVENTS, e({"type": "x"}, "cam-1", event_id="ev1"))
        msg = ws.receive_json()
        assert msg["type"] == "envelope"
        assert msg["data"]["event_id"] == "ev1"
        assert msg["data"]["topic"] == "events"


def test_ws_sin_filtro_recibe_todos_los_topicos():
    tap, view = make_view()
    client = TestClient(create_app(view))
    with client.websocket_connect("/ws") as ws:
        assert len(ws.receive_json()["topics"]) == len(list(Topic))
        tap.record(Topic.TRACKS, e({"a": 1}, event_id="t1"))
        assert ws.receive_json()["data"]["event_id"] == "t1"


def test_ws_topico_invalido_se_rechaza():
    _, view = make_view()
    client = TestClient(create_app(view))
    with pytest.raises(Exception), client.websocket_connect("/ws?topics=nope") as ws:  # noqa: B017
        ws.receive_json()


def test_ws_filtro_por_stream():
    tap, view = make_view()
    client = TestClient(create_app(view))
    with client.websocket_connect("/ws?topics=events&stream_id=cam-2") as ws:
        ws.receive_json()
        tap.record(Topic.EVENTS, e({"t": 1}, "cam-1", event_id="a"))
        tap.record(Topic.EVENTS, e({"t": 1}, "cam-2", event_id="b"))
        assert ws.receive_json()["data"]["event_id"] == "b"


def test_ws_desconexion_quita_el_oyente():
    tap, view = make_view()
    client = TestClient(create_app(view))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert client.get("/health").json()["ws_clients"] == 1
    assert client.get("/health").json()["ws_clients"] == 0
    tap.record(Topic.EVENTS, e({"t": 1}))  # no explota sin clientes


# --- Seguridad ---


def test_bind_no_local_sin_token_se_niega_a_arrancar():
    _, view = make_view()
    for host in ("0.0.0.0", "192.168.1.5", "example.com", "::"):
        with pytest.raises(ConfigurationError, match="token"):
            create_app(view, host=host)
    create_app(view, host="0.0.0.0", token="s3cret")
    for host in ("127.0.0.1", "localhost", "::1"):
        create_app(view, host=host)


def test_token_requerido_en_http_y_ws():
    _, view = make_view()
    client = TestClient(create_app(view, token="s3cret"))
    assert client.get("/agents").status_code == 401
    assert client.get("/agents", headers={"Authorization": "Bearer mal"}).status_code == 401
    assert client.get("/agents", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/agents", headers={"X-API-Token": "s3cret"}).status_code == 200
    with pytest.raises(Exception), client.websocket_connect("/ws") as ws:  # noqa: B017
        ws.receive_json()
    with client.websocket_connect("/ws?token=s3cret") as ws:
        assert ws.receive_json()["type"] == "hello"
    with client.websocket_connect("/ws", headers={"Authorization": "Bearer s3cret"}) as ws:
        assert ws.receive_json()["type"] == "hello"
