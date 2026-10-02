"""Endpoints /traces de la API de observabilidad (solo lectura)."""

from starlette.testclient import TestClient

from bus import InMemoryHub, MetadataEnvelope, Topic
from comms.api import create_app
from comms.tap import BusTap
from comms.view import TapSystemView
from tracing import TraceStore


def build():
    tap = BusTap(InMemoryHub())
    store = TraceStore()
    tap.add_listener(store.record)
    det = MetadataEnvelope(source="inference", payload={"detections": []}, event_id="c1", stream_id="cam-1")
    trk = det.derive("tracker", {"tracks": []}, suffix="tracks")
    evt = trk.derive("event", {"type": "zone_intrusion", "zone_id": "z1"}, suffix="events/0")
    tap.record(Topic.DETECTIONS, det)
    tap.record(Topic.TRACKS, trk)
    tap.record(Topic.EVENTS, evt)
    return TestClient(create_app(TapSystemView(tap), traces=store, token="t")), det


H = {"x-api-token": "t"}


def test_lista_de_trazas():
    client, det = build()
    body = client.get("/traces?limit=5", headers=H).json()
    (row,) = body["traces"]
    assert row["correlation_id"] == det.correlation_id
    assert row["hops"] == 3 and row["has_alert"] is True


def test_traza_por_correlacion_devuelve_la_cadena():
    client, det = build()
    body = client.get(f"/traces/{det.correlation_id}", headers=H).json()
    assert [h["topic"] for h in body["hops"]] == ["vision.detections", "vision.tracks", "events"]
    assert body["hops"][1]["causation_id"] == det.event_id
    assert body["end_to_end_ms"] is not None


def test_traza_inexistente_es_404():
    client, _ = build()
    assert client.get("/traces/nope", headers=H).status_code == 404


def test_traces_exige_token_y_valida_limit():
    client, _ = build()
    assert client.get("/traces").status_code == 401
    assert client.get("/traces?limit=0", headers=H).status_code == 422


def test_stop_no_bloquea_el_loop_si_langfuse_no_responde():
    import asyncio
    import time

    from comms import ObservabilityCommsHandler

    class Hanging:
        closed = False

        def on_message(self, topic, data):
            pass

        def close(self):
            time.sleep(1.5)  # host inalcanzable: flush/shutdown se cuelgan
            self.closed = True

    async def run():
        handler = ObservabilityCommsHandler(InMemoryHub(), port=0, close_timeout=0.2)
        handler._tracer = Hanging()
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        hb = asyncio.create_task(heartbeat())
        t0 = time.perf_counter()
        await handler.stop()
        elapsed = time.perf_counter() - t0
        hb.cancel()
        return elapsed, ticks

    elapsed, ticks = asyncio.run(run())
    assert elapsed < 1.0  # respetó el timeout
    assert ticks >= 5  # el loop siguió vivo mientras tanto


def test_sin_almacen_no_hay_rutas_de_trazas():
    tap = BusTap(InMemoryHub())
    client = TestClient(create_app(TapSystemView(tap)))
    assert client.get("/traces").status_code == 404


def test_el_handler_cablea_el_almacen_al_tap():
    import asyncio

    from comms import ObservabilityCommsHandler

    async def run():
        handler = ObservabilityCommsHandler(InMemoryHub(), port=0)
        env = MetadataEnvelope(source="event", payload={"type": "x"}, event_id="solo")
        handler.tap.record(Topic.EVENTS, env)
        return handler.traces.get("solo")

    trace = asyncio.run(run())
    assert trace is not None and trace["has_alert"] is True
