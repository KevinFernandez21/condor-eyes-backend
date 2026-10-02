"""Simuladores integrados del perfil ``sim`` y de la degradación en ``laptop``.

Publican **las mismas formas de payload** que los módulos reales (ubicación de
#32/#37, identidad de #36, actuación de #33) para que el dashboard muestre todo
hoy, sin depender de código aún no integrado. Solo metadata: JSON estricto, sin
imágenes ni embeddings.

Mientras ``Topic.LOCATION`` (#32) no exista en ``main``, las estimaciones de
ubicación y los resultados de identidad viajan por ``Topic.EVENTS`` con un
campo ``kind`` que los distingue (``location.estimate`` / ``identity.result``).
Cuando #32 se integre, ``LOCATION_TOPIC`` pasa a ser ``Topic.LOCATION`` sin
tocar nada más.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bus import MetadataEnvelope, MetadataHub, Topic

from .config import SystemConfig

logger = logging.getLogger(__name__)

LOCATION_TOPIC: Topic = getattr(Topic, "LOCATION", Topic.EVENTS)
"""Tópico de las estimaciones de ubicación (``Topic.LOCATION`` si ya existe)."""

IDENTITY_TOPIC: Topic = Topic.EVENTS
"""Los resultados de identidad aún no tienen tópico propio: usan ``EVENTS``."""

LOCATION_KIND = "location.estimate"
IDENTITY_KIND = "identity.result"
DEFAULT_PEOPLE = ("person-sim-01", "person-sim-02")

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


# --- payloads ---------------------------------------------------------------


def location_payload(
    person_ref: str,
    zone_id: str,
    confidence: float,
    now: datetime,
    *,
    authorized: bool | None = None,
    node_id: str = "N0001",
) -> dict[str, Any]:
    """Estimación ``located`` con la forma de ``location.bus_adapter.to_envelope``."""
    stamp = now.isoformat()
    return {
        "kind": LOCATION_KIND,
        "simulated": True,
        "tag_ref": f"tag-{person_ref}",
        "person_ref": person_ref,
        "status": "located",
        "zone_id": zone_id,
        "confidence": confidence,
        "unknown_reason": None,
        "degraded": False,
        "authorized": authorized,
        "battery_pct": 90,
        "battery_low": False,
        "first_evidence_at": stamp,
        "last_evidence_at": stamp,
        "computed_at": stamp,
        "evidence": [
            {
                "node_id": node_id,
                "zone_id": zone_id,
                "smoothed_rssi_dbm": -59.0,
                "samples": 5,
                "first_seen_at": stamp,
                "last_seen_at": stamp,
            }
        ],
    }


def unknown_location_payload(reason: str, now: datetime) -> dict[str, Any]:
    """Estimación ``unknown``: no inventa zona ni persona (sin C6, sin señal)."""
    return {
        "kind": LOCATION_KIND,
        "simulated": True,
        "tag_ref": None,
        "person_ref": None,
        "status": "unknown",
        "zone_id": None,
        "confidence": 0.0,
        "unknown_reason": reason,
        "degraded": True,
        "authorized": None,
        "battery_pct": None,
        "battery_low": False,
        "first_evidence_at": None,
        "last_evidence_at": None,
        "computed_at": now.isoformat(),
        "evidence": [],
    }


def identity_payload(
    track_ref: str, person_id: str, score: float, *, observed_at: datetime | None = None
) -> dict[str, Any]:
    """Resultado ``match`` con la forma de ``faceid.identity`` (sin embedding)."""
    when = observed_at or _utc_now()
    return {
        "kind": IDENTITY_KIND,
        "simulated": True,
        "track_ref": track_ref,
        "status": "match",
        "person_id": person_id,
        "score": score,
        "threshold": 0.65,
        "requires_operator": True,
        "model_id": "sim",
        "observed_at": when.isoformat(),
        "evidence": {"enrolled": len(DEFAULT_PEOPLE)},
    }


def ptz_command_payload(
    camera_id: str, *, seq: int, pan: float, tilt: float, speed: float
) -> dict[str, Any]:
    return {
        "kind": "ptz.command",
        "seq": seq,
        "camera_id": camera_id,
        "pan_deg": pan,
        "tilt_deg": tilt,
        "speed_dps": speed,
        "state": "tracking",
        "simulated": True,
    }


def ptz_health_payload(camera_id: str, *, acked: int) -> dict[str, Any]:
    return {
        "kind": "ptz.health",
        "camera_id": camera_id,
        "link": "ok",
        "moves_sent": acked,
        "acks": acked,
        "timeouts": 0,
        "simulated": True,
    }


# --- detector sintético -----------------------------------------------------


class MovingDetector:
    """Detector sintético (cumple ``Detector``): dos personas, una cruza el encuadre."""

    name = "moving-sim"

    def __init__(self, width: int = 640, height: int = 480, step: float = 8.0) -> None:
        self._width = width
        self._height = height
        self._step = step
        self._frame = 0

    def warmup(self, n: int = 5) -> None:
        return None

    def infer(self, frame: Any) -> list[dict]:
        w, h = float(self._width), float(self._height)
        box_w, box_h = 60.0, 160.0
        span = w - box_w
        phase = (self._frame * self._step) % (2 * span)
        x = phase if phase <= span else 2 * span - phase  # va y vuelve
        self._frame += 1
        top = h * 0.4
        return [
            {"xyxy": [x, top, x + box_w, top + box_h], "cls": 0, "conf": 0.9},
            {"xyxy": [60.0, top, 60.0 + box_w, top + box_h], "cls": 0, "conf": 0.8},
        ]

    def close(self) -> None:
        return None


# --- componentes periódicos -------------------------------------------------


class PeriodicComponent:
    """Base: tareas asyncio con arranque/parada idempotente y salud en metadata."""

    name = "component"

    def __init__(
        self,
        hub: MetadataHub,
        interval_s: float,
        *,
        status: str = "simulated",
        detail: str = "",
    ) -> None:
        self._hub = hub
        self._interval = interval_s
        self._status = status
        self._detail = detail
        self._tasks: list[asyncio.Task[None]] = []
        self._started = False
        self.published = 0
        self.errors = 0
        self.last_error: str | None = None

    async def start(self) -> None:
        if self._started:
            raise RuntimeError(f"El componente '{self.name}' ya está iniciado")
        self._started = True
        try:
            await self._on_start()
        except BaseException:
            self._started = False
            raise
        self._spawn(self._loop())

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        await self._on_stop()
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _on_start(self) -> None:
        """Gancho para suscripciones; por defecto nada."""

    async def _on_stop(self) -> None:
        """Gancho para cerrar suscripciones; por defecto nada."""

    def _spawn(self, coro: Any) -> None:
        self._tasks.append(asyncio.create_task(coro, name=f"system:{self.name}"))

    async def _loop(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - un tick fallido no mata el componente
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("'%s' falló en un ciclo: %s", self.name, self.last_error)
            await asyncio.sleep(self._interval)

    async def _tick(self) -> None:  # pragma: no cover - lo define cada componente
        raise NotImplementedError

    async def _publish(
        self,
        topic: Topic,
        source: str,
        payload: Mapping[str, Any],
        stream_id: str | None = None,
    ) -> None:
        await self._hub.publish(
            topic, MetadataEnvelope(source=source, payload=payload, stream_id=stream_id)
        )
        self.published += 1

    def health(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "detail": self._detail,
            "published": self.published,
            "errors": self.errors,
            "last_error": self.last_error,
        }


def _people(cfg: SystemConfig) -> tuple[str, ...]:
    return tuple(cfg.permissions) or DEFAULT_PEOPLE


class LocationSimulator(PeriodicComponent):
    """Tag simulado: cada persona permanece en una zona. Con ``degraded_reason``
    (p. ej. sin C6) solo publica ``unknown``: la ubicación nunca se inventa."""

    name = "location"

    def __init__(
        self,
        hub: MetadataHub,
        cfg: SystemConfig,
        *,
        degraded_reason: str | None = None,
        detail: str = "",
        clock: Clock = _utc_now,
    ) -> None:
        super().__init__(
            hub,
            cfg.tag.interval_s,
            status="degraded" if degraded_reason else "simulated",
            detail=detail or (degraded_reason or "tag simulado"),
        )
        self._cfg = cfg
        self._reason = degraded_reason
        self._clock = clock

    @property
    def topic(self) -> Topic:
        return LOCATION_TOPIC

    def _payloads(self) -> list[dict[str, Any]]:
        now = self._clock()
        if self._reason:
            return [unknown_location_payload(self._reason, now)]
        zones = [z.zone_id for z in self._cfg.zones] or ["zone-1"]
        out = []
        for index, person in enumerate(_people(self._cfg)):
            zone = zones[(len(zones) - 1 - index) % len(zones)]
            allowed = self._cfg.permissions.get(person)
            out.append(
                location_payload(
                    person,
                    zone,
                    0.9,
                    now,
                    authorized=None if allowed is None else zone in allowed,
                    node_id=f"N{index + 1:04d}",
                )
            )
        return out

    async def _tick(self) -> None:
        for payload in self._payloads():
            await self._publish(self.topic, "location", payload)


class TagReplay(LocationSimulator):
    """Reproduce estimaciones grabadas (JSONL) con sus ``offset_s``; en bucle.

    Cada línea es un payload parcial (``person_ref``, ``zone_id``,
    ``confidence`` y opcional ``offset_s``). Si el archivo falta o no tiene
    líneas válidas, degrada a ``unknown`` en vez de caerse.
    """

    def __init__(
        self, hub: MetadataHub, cfg: SystemConfig, *, clock: Clock = _utc_now
    ) -> None:
        self._entries, problem = self._load(cfg.tag.path)
        super().__init__(
            hub,
            cfg,
            degraded_reason="tag_missing" if problem else None,
            detail=problem or f"{len(self._entries)} estimaciones de {cfg.tag.path}",
            clock=clock,
        )
        if not problem:
            self._status = "ok"
        self._cursor = 0
        self._origin: float | None = None

    @staticmethod
    def _load(path: str) -> tuple[list[dict[str, Any]], str]:
        if not path or not Path(path).is_file():
            return [], f"JSONL de tags no encontrado: {path or '(sin ruta)'}"
        entries: list[dict[str, Any]] = []
        for number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                offset = float(item.get("offset_s", 0.0))
                if not (
                    isinstance(item, dict)
                    and math.isfinite(offset)
                    and isinstance(item["person_ref"], str)
                    and isinstance(item["zone_id"], str)
                ):
                    raise ValueError("estimación incompleta")
                confidence = float(item.get("confidence", 0.9))
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("confianza fuera de [0, 1]")
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                logger.warning("%s:%d ignorada: %s", path, number, exc)
                continue
            entries.append(
                {
                    "offset_s": offset,
                    "person_ref": item["person_ref"],
                    "zone_id": item["zone_id"],
                    "confidence": confidence,
                }
            )
        if not entries:
            return [], f"JSONL de tags sin líneas válidas: {path}"
        return sorted(entries, key=lambda e: e["offset_s"]), ""

    def _payloads(self) -> list[dict[str, Any]]:
        if self._reason or not self._entries:
            return super()._payloads()
        loop_time = asyncio.get_running_loop().time()
        if self._origin is None:
            self._origin = loop_time
        elapsed = loop_time - self._origin
        out: list[dict[str, Any]] = []
        now = self._clock()
        while self._cursor < len(self._entries) and (
            self._entries[self._cursor]["offset_s"] <= elapsed
        ):
            e = self._entries[self._cursor]
            out.append(
                location_payload(e["person_ref"], e["zone_id"], e["confidence"], now)
            )
            self._cursor += 1
        if (
            self._cursor >= len(self._entries)
            and elapsed > self._entries[-1]["offset_s"]
        ):
            self._cursor, self._origin = 0, loop_time  # bucle
        return out


class IdentitySimulator(PeriodicComponent):
    """Reconocimiento facial simulado sobre los tracks que circulan por el bus."""

    name = "identity"

    def __init__(self, hub: MetadataHub, cfg: SystemConfig) -> None:
        super().__init__(hub, cfg.identity.interval_s, detail="identidad simulada")
        self._cfg = cfg
        self._seen: dict[str, tuple[str | None, float]] = {}
        self._subscription: Any = None

    @property
    def topic(self) -> Topic:
        return IDENTITY_TOPIC

    async def _on_start(self) -> None:
        self._subscription = self._hub.subscribe(Topic.TRACKS)
        self._spawn(self._consume())

    async def _on_stop(self) -> None:
        if self._subscription is not None:
            self._subscription.close()
            self._subscription = None

    async def _consume(self) -> None:
        subscription = self._subscription
        loop = asyncio.get_running_loop()
        async for envelope in subscription:
            for track in envelope.payload.get("tracks", []):
                ref = track.get("track_ref")
                if isinstance(ref, str):
                    self._seen[ref] = (envelope.stream_id, loop.time())

    async def _tick(self) -> None:
        now = asyncio.get_running_loop().time()
        people = _people(self._cfg)
        for ref, (stream_id, seen_at) in list(self._seen.items()):
            if now - seen_at > 2.0:
                del self._seen[ref]
                continue
            tail = ref.rsplit("/", 1)[-1]
            number = int(tail) if tail.isdigit() else 1
            person = people[(number - 1) % len(people)]
            await self._publish(
                self.topic,
                "identity",
                identity_payload(ref, person, 0.9),
                stream_id=stream_id,
            )


class ActuatorSimulator(PeriodicComponent):
    """Actuador pan-tilt simulado: comandos con ack inmediato y salud del enlace."""

    name = "actuation"

    def __init__(self, hub: MetadataHub, cfg: SystemConfig) -> None:
        super().__init__(hub, cfg.actuator.interval_s, detail="pan-tilt simulado")
        self._camera = cfg.camera.stream_id
        self._seq = 0

    async def _tick(self) -> None:
        self._seq += 1
        pan = 30.0 * math.sin(self._seq / 5.0)
        tilt = 10.0 * math.cos(self._seq / 7.0)
        await self._publish(
            Topic.EVENTS,
            "actuation",
            ptz_command_payload(
                self._camera, seq=self._seq, pan=pan, tilt=tilt, speed=20.0
            ),
            stream_id=self._camera,
        )
        await self._publish(
            Topic.HEALTH,
            "actuation",
            ptz_health_payload(self._camera, acked=self._seq),
            stream_id=self._camera,
        )
