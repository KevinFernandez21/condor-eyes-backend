"""Pruebas del adaptador en memoria del bus (determinista, sin AgentScope)."""

import asyncio

import pytest

from bus import (
    InvalidEnvelopeError,
    InvalidTopicError,
    MetadataEnvelope,
    Topic,
    UnsupportedVersionError,
)
from bus.memory import HubClosedError, InMemoryHub


def env(event_id: str = "e1", **kwargs) -> MetadataEnvelope:
    return MetadataEnvelope(
        source="tracker", payload={"n": 1}, event_id=event_id, **kwargs
    )


async def drain(subscription) -> list[MetadataEnvelope]:
    return [message async for message in subscription]


async def test_publish_fans_out_to_every_subscriber_of_the_topic():
    hub = InMemoryHub()
    a = hub.subscribe(Topic.EVENTS)
    b = hub.subscribe(Topic.EVENTS)
    other = hub.subscribe(Topic.TRACKS)

    await hub.publish(Topic.EVENTS, env("e1"))
    await hub.close()

    assert [m.event_id for m in await drain(a)] == ["e1"]
    assert [m.event_id for m in await drain(b)] == ["e1"]
    assert await drain(other) == []


async def test_messages_keep_publication_order():
    hub = InMemoryHub()
    sub = hub.subscribe(Topic.EVENTS)
    for i in range(5):
        await hub.publish(Topic.EVENTS, env(f"e{i}"))
    await hub.close()
    assert [m.event_id for m in await drain(sub)] == [f"e{i}" for i in range(5)]


async def test_publish_rejects_invalid_topic_and_versions_with_clear_errors():
    hub = InMemoryHub()
    with pytest.raises(InvalidTopicError, match="Topic desconocido"):
        await hub.publish("video.frames", env())  # type: ignore[arg-type]
    with pytest.raises(UnsupportedVersionError, match="payload_version"):
        await hub.publish(Topic.EVENTS, env(payload_version=9))
    with pytest.raises(InvalidEnvelopeError, match="binarios"):
        await hub.publish(
            Topic.EVENTS, MetadataEnvelope(source="x", payload={"frame": b"raw"})
        )
    assert hub.stats["published"] == 0


async def test_subscribe_rejects_invalid_topic():
    hub = InMemoryHub()
    with pytest.raises(InvalidTopicError):
        hub.subscribe("nope")  # type: ignore[arg-type]


async def test_close_drains_pending_messages_then_ends_iteration():
    hub = InMemoryHub()
    sub = hub.subscribe(Topic.HEALTH)
    await hub.publish(Topic.HEALTH, env("h1"))
    await hub.close()
    assert [m.event_id for m in await drain(sub)] == ["h1"]


async def test_publish_after_close_fails():
    hub = InMemoryHub()
    await hub.close()
    with pytest.raises(HubClosedError, match="cerrado"):
        await hub.publish(Topic.EVENTS, env())
    with pytest.raises(HubClosedError):
        hub.subscribe(Topic.EVENTS)


async def test_telemetry_topics_drop_oldest_and_report_error_event():
    hub = InMemoryHub(queue_size=2)
    sub = hub.subscribe(Topic.DETECTIONS)
    errors = hub.subscribe(Topic.ERRORS)
    for i in range(4):
        await hub.publish(Topic.DETECTIONS, env(f"d{i}"))
    await hub.close()
    assert [m.event_id for m in await drain(sub)] == ["d2", "d3"]
    assert hub.stats["dropped"] == 2
    (first,) = await drain(errors)  # throttled: solo el primer descarte
    assert first.source == "hub"
    assert first.payload["stage"] == "queue_overflow"
    assert first.payload["failed_topic"] == "vision.detections"
    assert first.payload["failed_event_id"] == "d0"


async def test_overflow_error_events_are_throttled_every_100_drops():
    hub = InMemoryHub(queue_size=5)
    hub.subscribe(Topic.DETECTIONS)
    errors = hub.subscribe(Topic.ERRORS)
    for i in range(206):
        await hub.publish(Topic.DETECTIONS, env(f"d{i}"))
    await hub.close()
    reported = await drain(errors)
    assert [m.payload["dropped_total"] for m in reported] == [1, 101, 201]


async def test_events_and_commands_are_lossless_beyond_queue_size():
    hub = InMemoryHub(queue_size=2)
    sub = hub.subscribe(Topic.EVENTS)
    errors = hub.subscribe(Topic.ERRORS)
    for i in range(5):
        await hub.publish(Topic.EVENTS, env(f"e{i}"))
    await hub.close()
    assert [m.event_id for m in await drain(sub)] == [f"e{i}" for i in range(5)]
    assert hub.stats["dropped"] == 0
    assert hub.stats["overflow"] == 3
    (report,) = await drain(errors)
    assert report.payload["stage"] == "queue_overflow"
    assert report.payload["policy"] == "lossless"


async def test_error_topic_overflow_does_not_recurse():
    hub = InMemoryHub(queue_size=1)
    hub.subscribe(Topic.ERRORS)
    for i in range(3):
        await hub.publish(Topic.ERRORS, env(f"x{i}"))
    assert hub.stats["dropped"] == 2
    assert hub.stats["published"] == 3


async def test_history_is_bounded_and_can_be_disabled():
    hub = InMemoryHub(history_size=2)
    for i in range(4):
        await hub.publish(Topic.HEALTH, env(f"h{i}"))
    assert [m.event_id for _, m in hub.history] == ["h2", "h3"]
    off = InMemoryHub(history_size=0)
    await off.publish(Topic.HEALTH, env("h"))
    assert list(off.history) == []


async def test_subscription_close_stops_only_that_subscriber():
    hub = InMemoryHub()
    a = hub.subscribe(Topic.EVENTS)
    b = hub.subscribe(Topic.EVENTS)
    a.close()
    await hub.publish(Topic.EVENTS, env("e1"))
    await hub.close()
    assert await drain(a) == []
    assert [m.event_id for m in await drain(b)] == ["e1"]


async def test_consumer_waiting_is_woken_by_publish():
    hub = InMemoryHub()
    sub = hub.subscribe(Topic.EVENTS)
    waiter = asyncio.create_task(sub.__anext__())
    await asyncio.sleep(0)
    await hub.publish(Topic.EVENTS, env("e1"))
    assert (await asyncio.wait_for(waiter, 1)).event_id == "e1"
    await hub.close()


async def test_history_records_published_messages_for_assertions():
    hub = InMemoryHub()
    await hub.publish(Topic.EVENTS, env("e1"))
    assert [(t, m.event_id) for t, m in hub.history] == [(Topic.EVENTS, "e1")]
