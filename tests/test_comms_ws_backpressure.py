"""Un cliente WebSocket lento o muerto nunca frena el bus."""

import asyncio
import time

from bus import InMemoryHub, MetadataEnvelope, Topic
from comms.api import ClientChannel
from comms.tap import BusTap


def env(i):
    return MetadataEnvelope(source="x", payload={"i": i}, event_id=f"e{i}")


async def test_canal_descarta_el_mas_antiguo_y_cuenta():
    ch = ClientChannel(max_queue=3)
    for i in range(10):
        ch.push({"n": i})
    assert ch.dropped == 7
    assert ch.queued == 3


async def test_cliente_que_no_lee_no_bloquea_el_bus():
    hub = InMemoryHub(queue_size=100_000)
    tap = BusTap(hub)
    channel = ClientChannel(max_queue=8)
    tap.add_listener(lambda topic, data: channel.push(data))
    release = asyncio.Event()

    async def stuck_send(message):  # simula un socket con el buffer lleno
        await release.wait()

    sender = asyncio.create_task(channel.run(stuck_send))
    await tap.start()
    start = time.monotonic()
    for i in range(5_000):
        await asyncio.wait_for(hub.publish(Topic.EVENTS, env(i)), timeout=1.0)
    await asyncio.sleep(0.05)
    assert time.monotonic() - start < 5.0
    assert channel.queued <= 8
    assert channel.dropped >= 4_000
    assert len(tap.events(limit=10)) == 10  # el tap sigue al día
    release.set()
    channel.close()
    await sender
    await tap.stop()


async def test_cliente_que_falla_al_enviar_se_cierra_y_no_afecta_al_resto():
    hub = InMemoryHub()
    tap = BusTap(hub)
    dead = ClientChannel(max_queue=4)
    alive = ClientChannel(max_queue=4)
    got = []

    async def broken(message):
        raise ConnectionError("socket cerrado")

    async def fine(message):
        got.append(message)

    remove_dead = tap.add_listener(lambda t, d: dead.push(d))
    tap.add_listener(lambda t, d: alive.push(d))
    dead_task = asyncio.create_task(dead.run(broken, on_close=remove_dead))
    alive_task = asyncio.create_task(alive.run(fine))
    await tap.start()
    for i in range(20):
        await hub.publish(Topic.EVENTS, env(i))
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)
    assert dead.closed
    await dead_task
    assert len(got) >= 1
    alive.close()
    await alive_task
    await tap.stop()


async def test_lag_se_notifica_al_cliente():
    ch = ClientChannel(max_queue=2)
    for i in range(5):
        ch.push({"n": i})
    out = []

    async def send(message):
        out.append(message)

    task = asyncio.create_task(ch.run(send))
    await asyncio.sleep(0.01)
    ch.close()
    await task
    assert out[0] == {"type": "lag", "dropped": 3, "dropped_total": 3}
    assert [m["data"]["n"] for m in out[1:]] == [3, 4]
    assert all(m["type"] == "envelope" for m in out[1:])
