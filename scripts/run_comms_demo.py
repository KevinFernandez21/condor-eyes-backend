"""Demo de la API de observabilidad: runtime + InMemoryHub + publicador sintético.

Uso (desde la raíz del repo)::

    uv run python scripts/run_comms_demo.py            # sirve hasta Ctrl+C
    uv run python scripts/run_comms_demo.py --seconds 20

Abre http://127.0.0.1:8000/docs para probar los endpoints y, por ejemplo,
``ws://127.0.0.1:8000/ws?topics=events`` para el flujo vivo. Todo es metadata
sintética: no hay frames ni cámaras reales.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.handlers import build_default_handlers
from agents.runtime import AgentRuntime
from bus import InMemoryHub, Topic
from comms import ConfigurationError, ObservabilityCommsHandler

CAMERAS = ("cam-1", "cam-2")
ZONES = ("entrada", "bodega", "perimetro")


class _Sink:
    async def save(self, envelope) -> None:  # EventSink
        pass

    async def send(self, envelope) -> None:  # AlertSink
        pass


def _event_rule(track):
    """Un evento por cada mensaje de tracks, con zona sintética."""
    zone = ZONES[hash(track.event_id) % len(ZONES)]
    return [{"type": "person_in_zone", "zone_id": zone, "tracks": len(track.payload["tracks"])}]


async def _publish_synthetic(runtime: AgentRuntime, period: float) -> None:
    rng = random.Random(7)
    n = 0
    for camera in CAMERAS:
        await runtime.emit("ingest", Topic.STREAM_STATUS, {"state": "up"}, stream_id=camera)
    while True:
        n += 1
        camera = CAMERAS[n % len(CAMERAS)]
        detections = [
            {"id": i, "label": "person", "confidence": round(rng.uniform(0.5, 0.99), 2)}
            for i in range(rng.randint(1, 3))
        ]
        det = await runtime.emit(
            "inference", Topic.DETECTIONS, {"detections": detections}, stream_id=camera
        )
        if n % 5 == 0:  # decisión de fusión sintética; identificadores ya seudonimizados
            zone = rng.choice(ZONES)
            await runtime.emit(
                "event",
                Topic.EVENTS,
                {
                    "decision_id": f"dec-{n:04d}",
                    "outcome": rng.choice(["corroborated", "uncorroborated"]),
                    "confidence": round(rng.uniform(0.4, 0.95), 2),
                    "zone_id": zone,
                    "stream_id": camera,
                    "person_id": f"p-{rng.randrange(16**4):04x}",
                    "requires_operator": True,
                    # el tracker deriva su event_id de forma determinista: <detección>/tracks
                    "evidence": [
                        {
                            "evidence_id": f"{det.event_id}/tracks",
                            "kind": "track",
                            "role": "supports",
                            "stream_id": camera,
                        }
                    ],
                },
                stream_id=camera,
            )
        await asyncio.sleep(period)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default=None, help="obligatorio si el host no es local")
    parser.add_argument("--seconds", type=float, default=0, help="0 = hasta Ctrl+C")
    parser.add_argument("--period", type=float, default=0.2, help="segundos entre mensajes")
    args = parser.parse_args()

    hub = InMemoryHub()
    try:
        comms = ObservabilityCommsHandler(hub, host=args.host, port=args.port, token=args.token)
    except ConfigurationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    handlers = build_default_handlers(
        storage_sink=_Sink(), alert_sink=_Sink(), event_rules=[_event_rule]
    )
    handlers["comms"] = comms
    runtime = AgentRuntime(hub, handlers, heartbeat_interval=1.0)
    comms.bind_runtime(runtime)
    await runtime.start()
    print(f"API de observabilidad en http://{args.host}:{comms.port}/docs", flush=True)
    publisher = asyncio.create_task(_publish_synthetic(runtime, args.period))
    try:
        await asyncio.sleep(args.seconds) if args.seconds else await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        publisher.cancel()
        await asyncio.gather(publisher, return_exceptions=True)
        await runtime.stop()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
