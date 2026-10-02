"""Pruebas del tap del bus: buffers acotados, estadísticas y oyentes."""

import asyncio
from datetime import UTC, datetime

import pytest

from bus import InMemoryHub, MetadataEnvelope, Topic
from comms.tap import BusTap


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def env(source="x", payload=None, stream_id=None, event_id=None):
    kwargs = {"event_id": event_id} if event_id else {}
    return MetadataEnvelope(
        source=source, payload=payload or {"k": 1}, stream_id=stream_id, **kwargs
    )


async def settle(n: int = 20) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def test_record_serializa_y_acota_los_buffers_por_topico():
    tap = BusTap(InMemoryHub(), topic_size=3)
    for i in range(5):
        tap.record(Topic.TRACKS, env(event_id=f"e{i}"))
    recent = tap.recent(Topic.TRACKS, limit=10)
    assert [m["event_id"] for m in recent] == ["e4", "e3", "e2"]  # más nuevo primero
    assert all(m["topic"] == "vision.tracks" for m in recent)


def test_eventos_y_decisiones_en_anillos_acotados():
    tap = BusTap(InMemoryHub(), events_size=4, decisions_size=2)
    for i in range(6):
        tap.record(Topic.EVENTS, env(payload={"type": "t", "n": i}, event_id=f"v{i}"))
    for i in range(3):
        tap.record(
            Topic.EVENTS,
            env(payload={"decision_id": f"d{i}", "outcome": "corroborated"}, event_id=f"d{i}"),
        )
    assert len(tap.events(limit=100)) == 4
    decisions = tap.decisions(limit=100)
    assert [d["payload"]["decision_id"] for d in decisions] == ["d2", "d1"]


def test_filtra_por_stream_y_zona():
    tap = BusTap(InMemoryHub())
    tap.record(Topic.EVENTS, env(payload={"decision_id": "a", "zone_id": "z1"}, stream_id="cam-1"))
    tap.record(Topic.EVENTS, env(payload={"decision_id": "b", "zone_id": "z2"}, stream_id="cam-2"))
    tap.record(Topic.EVENTS, env(payload={"decision_id": "c", "zone": "z1"}, stream_id="cam-2"))
    assert [d["payload"]["decision_id"] for d in tap.decisions(stream_id="cam-2")] == ["c", "b"]
    assert [d["payload"]["decision_id"] for d in tap.decisions(zone="z1")] == ["c", "a"]
    assert [d["payload"]["decision_id"] for d in tap.decisions(stream_id="cam-2", zone="z2")] == ["b"]


def test_tasa_en_ventana_deslizante():
    clock = FakeClock()
    tap = BusTap(InMemoryHub(), window_s=10.0, clock=clock)
    for _ in range(20):
        tap.record(Topic.DETECTIONS, env())
        clock.now += 0.5  # 2 msg/s durante 10 s
    stats = {s["topic"]: s for s in tap.topic_stats()}
    assert stats["vision.detections"]["count"] == 20
    assert stats["vision.detections"]["rate_per_s"] == pytest.approx(2.0, abs=0.2)
    clock.now += 60
    stats = {s["topic"]: s for s in tap.topic_stats()}
    assert stats["vision.detections"]["rate_per_s"] == 0.0
    assert stats["vision.detections"]["count"] == 20  # el total no se ventanea


def test_topic_stats_incluye_todos_los_topicos_y_ultimo_mensaje():
    wall = datetime(2026, 1, 1, tzinfo=UTC)
    tap = BusTap(InMemoryHub(), wall=lambda: wall)
    tap.record(Topic.HEALTH, env())
    stats = {s["topic"]: s for s in tap.topic_stats()}
    assert set(stats) == {t.value for t in Topic}
    assert stats["system.health"]["last_message_at"] == wall.isoformat()
    assert stats["events"]["last_message_at"] is None


def test_payload_no_finito_no_entra():
    tap = BusTap(InMemoryHub())
    with pytest.raises(ValueError):
        tap.record(Topic.TRACKS, MetadataEnvelope(source="x", payload={"v": float("nan")}))
    assert tap.recent(Topic.TRACKS, 10) == []


def test_heartbeats_y_reinicios_se_derivan_del_bus():
    tap = BusTap(InMemoryHub())
    tap.record(Topic.HEALTH, env(source="tracker", payload={"role": "tracker", "state": "running"}))
    tap.record(
        Topic.COMMANDS,
        env(source="supervisor", payload={"action": "restart", "target": "tracker"}),
    )
    tap.record(
        Topic.COMMANDS,
        env(source="supervisor", payload={"action": "restart", "target": "tracker"}),
    )
    assert "tracker" in tap.heartbeats()
    assert tap.restarts() == {"tracker": 2}


def test_drops_del_bus_se_cuentan_por_topico():
    tap = BusTap(InMemoryHub())
    tap.record(
        Topic.ERRORS,
        env(
            source="hub",
            payload={
                "error_type": "OverflowError",
                "message": "Cola saturada",
                "stage": "queue_overflow",
                "failed_topic": "vision.detections",
                "policy": "drop_oldest",
                "dropped_total": 1,
            },
        ),
    )
    stats = {s["topic"]: s for s in tap.topic_stats()}
    assert stats["vision.detections"]["drops"] == 1
    assert stats["system.errors"]["drops"] == 0


async def test_start_se_suscribe_a_todos_los_topicos_por_el_hub_tipado():
    hub = InMemoryHub()
    tap = BusTap(hub)
    await tap.start()
    for topic in Topic:
        await hub.publish(topic, env())
    await settle()
    assert all(s["count"] >= 1 for s in tap.topic_stats())
    await tap.stop()
    assert hub._subscriptions[Topic.EVENTS] == []  # sin suscripciones colgadas


async def test_oyentes_reciben_y_se_pueden_quitar():
    hub = InMemoryHub()
    tap = BusTap(hub)
    got = []
    remove = tap.add_listener(lambda topic, data: got.append((topic, data["event_id"])))
    await tap.start()
    await hub.publish(Topic.EVENTS, env(event_id="a"))
    await settle()
    remove()
    await hub.publish(Topic.EVENTS, env(event_id="b"))
    await settle()
    await tap.stop()
    assert got == [(Topic.EVENTS, "a")]


async def test_oyente_que_falla_no_rompe_el_tap():
    hub = InMemoryHub()
    tap = BusTap(hub)

    def boom(topic, data):
        raise RuntimeError("oyente roto")

    tap.add_listener(boom)
    await tap.start()
    await hub.publish(Topic.EVENTS, env(event_id="a"))
    await hub.publish(Topic.EVENTS, env(event_id="b"))
    await settle()
    assert len(tap.events(10)) == 2
    await tap.stop()
