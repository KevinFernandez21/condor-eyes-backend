"""Endurecimiento de la API: Origin, tokens no ASCII, docs, frames, concurrencia."""

import inspect
import threading

import pytest
from _comms_ws import ws_session
from starlette.testclient import TestClient

from bus import InMemoryHub, MetadataEnvelope, Topic
from comms.api import create_app
from comms.tap import BusTap
from comms.view import TapSystemView


def make_view():
    tap = BusTap(InMemoryHub())
    return tap, TapSystemView(tap)


def e(payload, stream_id=None, source="s", event_id=None):
    kw = {"event_id": event_id} if event_id else {}
    return MetadataEnvelope(source=source, payload=payload, stream_id=stream_id, **kw)


def denied(client, url, **kwargs):
    with pytest.raises(Exception), client.websocket_connect(url, **kwargs) as ws:  # noqa: B017
        ws.receive_json()


def test_ws_origin_hostil_se_rechaza_por_defecto():
    _, view = make_view()
    client = TestClient(create_app(view))
    denied(client, "/ws", headers={"Origin": "http://evil.example"})
    denied(client, "/ws", headers={"Origin": "http://localhost:9999"})  # puerto ajeno
    denied(client, "/ws", headers={"Origin": "null"})


def test_ws_origin_local_y_ausente_se_permiten():
    _, view = make_view()
    client = TestClient(create_app(view))  # TestClient sirve en el puerto 80
    for origin in ("http://localhost", "http://127.0.0.1", "http://[::1]", "http://localhost:80"):
        with ws_session(client, "/ws", headers={"Origin": origin}) as ws:
            assert ws.receive_json()["type"] == "hello"
    with ws_session(client, "/ws") as ws:  # cliente no navegador: sin Origin
        assert ws.receive_json()["type"] == "hello"


def test_ws_allowed_origins_configurable():
    _, view = make_view()
    client = TestClient(create_app(view, allowed_origins=["https://dash.example"]))
    with ws_session(client, "/ws", headers={"Origin": "https://dash.example"}) as ws:
        assert ws.receive_json()["type"] == "hello"
    denied(client, "/ws", headers={"Origin": "http://localhost"})


def test_token_no_ascii_no_provoca_500_ni_excepcion():
    _, view = make_view()
    client = TestClient(create_app(view, token="s3cret"))
    resp = client.get("/agents", headers={b"x-api-token": "sécret".encode("latin-1")})
    assert resp.status_code == 401
    denied(client, "/ws?token=sécret")
    client2 = TestClient(create_app(view, token="sécret"))
    resp = client2.get("/agents", headers={b"x-api-token": "sécret".encode()})
    assert resp.status_code == 200
    with ws_session(client2, "/ws?token=sécret") as ws:
        assert ws.receive_json()["type"] == "hello"


def test_ws_token_por_subprotocolo():
    _, view = make_view()
    client = TestClient(create_app(view, token="s3cret"))
    with ws_session(client, "/ws", subprotocols=["token.s3cret"]) as ws:
        assert ws.accepted_subprotocol == "token.s3cret"
        assert ws.receive_json()["type"] == "hello"
    denied(client, "/ws", subprotocols=["token.mal"])


def test_docs_y_openapi_deshabilitados_con_token_y_abiertos_sin_el():
    _, view = make_view()
    with_token = TestClient(create_app(view, token="s3cret"))
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert with_token.get(path).status_code == 404
    open_client = TestClient(create_app(view))
    assert open_client.get("/docs").status_code == 200
    assert open_client.get("/openapi.json").status_code == 200


def test_ws_tolera_frames_binarios_del_cliente():
    tap, view = make_view()
    client = TestClient(create_app(view))
    with ws_session(client, "/ws") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00\x01")
        ws.send_text("hola")
        tap.record(Topic.EVENTS, e({"t": 1}, event_id="tras-binario"))
        assert ws.receive_json()["data"]["event_id"] == "tras-binario"


def test_endpoints_http_corren_en_el_loop_y_no_en_hilos():
    _, view = make_view()
    app = create_app(view)
    paths = {"/health", "/agents", "/topics", "/events", "/decisions"}
    inner = [
        route
        for included in app.routes
        for route in getattr(getattr(included, "original_router", None), "routes", [])
    ]
    routes = [r for r in inner if getattr(r, "path", "") in paths]
    assert len(routes) == 5
    assert all(inspect.iscoroutinefunction(r.endpoint) for r in routes)


def test_lectura_concurrente_con_escritura_no_falla():
    tap = BusTap(InMemoryHub(), events_size=50)
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer():
        i = 0
        while not stop.is_set():
            tap.record(Topic.EVENTS, e({"decision_id": f"d{i}", "zone_id": "z"}))
            i += 1

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(3000):
            try:
                tap.events(limit=50, zone="z")
                tap.decisions(limit=50)
                tap.topic_stats()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                break
    finally:
        stop.set()
        thread.join()
    assert not errors, errors[:1]
