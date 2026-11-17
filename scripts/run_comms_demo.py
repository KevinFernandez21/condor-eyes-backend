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
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.handlers import build_default_handlers
from agents.runtime import AgentRuntime
from bus import InMemoryHub, Topic
from comms import ConfigurationError, ObservabilityCommsHandler
from fusion.decision import (
    DecisionOutcome,
    DecisionRecord,
    EvidenceKind,
    EvidenceRef,
    EvidenceRole,
    ReasonCode,
)

# id -> (nombre, escena). Metadata de cámara que el dashboard usa para la maqueta.
CAMERAS = {
    "cam-01": ("Entrada principal", "Entrada"),
    "cam-02": ("Estacionamiento", "Estacionamiento"),
    "cam-03": ("Perímetro norte", "Perímetro"),
    "cam-04": ("Bodega 1", "Bodega"),
}
FRAME = (1920, 1080)
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


_SCENARIOS = (
    (DecisionOutcome.CORROBORATED, (ReasonCode.EVIDENCE_CONSISTENT,), (
        (EvidenceKind.TRACK, EvidenceRole.SUPPORTS), (EvidenceKind.IDENTITY, EvidenceRole.SUPPORTS),
        (EvidenceKind.LOCATION, EvidenceRole.SUPPORTS), (EvidenceKind.PERMISSION, EvidenceRole.SUPPORTS),
    )),
    (DecisionOutcome.ALERT, (ReasonCode.PERSON_WITHOUT_TAG,), (
        (EvidenceKind.TRACK, EvidenceRole.SUPPORTS), (EvidenceKind.LOCATION, EvidenceRole.CONFLICTS),
    )),
    (DecisionOutcome.ALERT, (ReasonCode.ZONE_NOT_PERMITTED, ReasonCode.IDENTITY_TAG_MISMATCH), (
        (EvidenceKind.IDENTITY, EvidenceRole.CONFLICTS), (EvidenceKind.PERMISSION, EvidenceRole.CONTEXT),
    )),
    (DecisionOutcome.INCONCLUSIVE, (ReasonCode.STALE_SENSOR, ReasonCode.HIDDEN_FACE), (
        (EvidenceKind.TRACK, EvidenceRole.SUPPORTS), (EvidenceKind.LOCATION, EvidenceRole.STALE),
    )),
)  # fmt: skip


def _synthetic_decision(rng: random.Random, n: int, camera: str) -> DecisionRecord:
    """Decisión realista: razones y referencias de evidencia; IDs ya seudonimizados."""
    outcome, reasons, evidence = rng.choice(_SCENARIOS)
    now = datetime.now(UTC)
    zone = rng.choice(ZONES)
    refs = tuple(
        EvidenceRef(
            evidence_id=f"ev-{n:04d}-{i}",
            kind=kind,
            role=role,
            observed_at=now,
            confidence=round(rng.uniform(0.5, 0.98), 2),
            stream_id=camera,
            zone_id=zone,
            detail="sintético",
        )
        for i, (kind, role) in enumerate(evidence)
    )
    return DecisionRecord(
        decision_id=f"dec-{n:04d}",
        evaluated_at=now,
        outcome=outcome,
        confidence=round(rng.uniform(0.4, 0.95), 2),
        reason_codes=reasons,
        evidence=refs,
        track_ref=f"trk-{rng.randrange(16**4):04x}",
        stream_id=camera,
        zone_id=zone,
        person_id=f"p-{rng.randrange(16**4):04x}",
    )


def _spawn_actors(rng: random.Random, camera: str) -> list[dict]:
    """Personas (y vehículos en el estacionamiento) que se mueven en coordenadas de 1920x1080."""
    actors = []
    for i in range(rng.randint(1, 3)):
        actors.append({"cls": 0, "label": "person", "w": 90.0, "h": 230.0,
                       "x": rng.uniform(100, 1700), "y": rng.uniform(500, 780),
                       "vx": rng.choice((-1, 1)) * rng.uniform(15, 45)})
    if camera == "cam-02":
        for i in range(rng.randint(1, 2)):
            actors.append({"cls": 2, "label": "car", "w": 330.0, "h": 170.0,
                           "x": rng.uniform(100, 1400), "y": rng.uniform(560, 760),
                           "vx": rng.choice((-1, 1)) * rng.uniform(40, 90)})
    for number, actor in enumerate(actors, start=1):
        actor["track_id"] = number
    return actors


def _step(actors: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for a in actors:
        a["x"] += a["vx"] + rng.uniform(-4, 4)
        if a["x"] < 0 or a["x"] + a["w"] > FRAME[0]:
            a["vx"] = -a["vx"]
            a["x"] = min(max(a["x"], 0.0), FRAME[0] - a["w"])
        out.append({"xyxy": [round(a["x"], 1), round(a["y"], 1), round(a["x"] + a["w"], 1),
                             round(a["y"] + a["h"], 1)],
                    "cls": a["cls"], "label": a["label"], "conf": round(rng.uniform(0.62, 0.97), 2),
                    "track_id": a["track_id"]})
    return out


def _dropped(camera: str, seconds: float) -> str | None:
    """Cortes sintéticos: cam-04 se reconecta y cam-03 cae un rato cada minuto y medio."""
    if camera == "cam-04" and 40 <= seconds % 75 < 52:
        return "reconnecting"
    if camera == "cam-03" and 20 <= seconds % 90 < 30:
        return "failed"
    return None


async def _publish_synthetic(runtime: AgentRuntime, period: float) -> None:
    rng = random.Random(7)
    actors = {camera: _spawn_actors(rng, camera) for camera in CAMERAS}
    started = time.monotonic()
    n = 0
    next_status = 0.0
    while True:
        n += 1
        now = time.monotonic() - started
        for camera, (name, scene) in CAMERAS.items():
            drop = _dropped(camera, now)
            if now >= next_status:
                await runtime.emit(
                    "ingest",
                    Topic.STREAM_STATUS,
                    {"state": drop or "up", "name": name, "scene": scene,
                     "width": FRAME[0], "height": FRAME[1],
                     "last_error": "sin señal de la fuente" if drop else None},
                    stream_id=camera,
                )
            if drop == "failed":
                continue  # sin detecciones mientras la cámara está caída
            detections = [] if drop else _step(actors[camera], rng)
            await runtime.emit(
                "inference",
                Topic.DETECTIONS,
                {"width": FRAME[0], "height": FRAME[1], "detections": detections},
                stream_id=camera,
            )
        if now >= next_status:
            next_status = now + 2.0
        if n % 8 == 0:  # decisión de fusión sintética (misma forma que DecisionRecord)
            camera = rng.choice(list(CAMERAS))
            await runtime.emit(
                "event",
                Topic.EVENTS,
                _synthetic_decision(rng, n, camera).to_payload(),
                stream_id=camera,
            )
        await asyncio.sleep(period)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default=None, help="obligatorio si el host no es local")
    parser.add_argument("--seconds", type=float, default=0, help="0 = hasta Ctrl+C")
    parser.add_argument("--period", type=float, default=0.4, help="segundos entre mensajes")
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
