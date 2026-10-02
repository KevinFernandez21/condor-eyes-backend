"""Escenarios sintéticos etiquetados para medir el detector de eventos.

Cada escenario es una cámara fija durante `duration_s` con personas que siguen
un comportamiento con guion. Las trayectorias reales (sin ruido) generan las
etiquetas; las observaciones que ve el motor llevan ruido de posición, huecos
por oclusión, cambios de ID del tracker, pérdida de detecciones y vibración
de cámara. Las etiquetas salen del guion y de la geometría real, nunca del
propio motor de reglas.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from .engine import point_in_polygon
from .model import EventType, GroundTruthEvent, TrackObservation, Zone

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Perturbation:
    noise: float = 0.004
    dropout: float = 0.03
    occlusion_prob: float = 0.3
    occlusion_s: tuple[float, float] = (0.5, 1.8)
    id_switch_prob: float = 0.05
    shake_prob: float = 0.15
    shake_amp: float = 0.012


# Held-out más duro que train/val: más ruido, oclusiones y vibración.
TRAIN = Perturbation()
TEST = Perturbation(
    noise=0.007,
    dropout=0.06,
    occlusion_prob=0.45,
    occlusion_s=(0.5, 3.0),
    id_switch_prob=0.1,
    shake_prob=0.3,
    shake_amp=0.025,
)


@dataclass
class Person:
    pid: int
    behavior: str
    path: list[tuple[float, Point]]
    authorized: bool = False
    loiter: tuple[float, float] | None = None  # (inicio reportable, fin) por guion


@dataclass
class Scenario:
    seed: int
    stream_id: str
    duration_s: float
    observations: list[TrackObservation]
    ground_truth: list[GroundTruthEvent]
    behaviors: dict[str, int] = field(default_factory=dict)
    # track_id -> comportamiento y perturbaciones sufridas, para el análisis de fallos.
    track_info: dict[int, dict] = field(default_factory=dict)
    shakes: list[tuple[float, float]] = field(default_factory=list)


class _Scene:
    def __init__(self, zones: Sequence[Zone], rng: random.Random, dt: float) -> None:
        self.rng, self.dt = rng, dt
        self.restricted = next(z for z in zones if z.restricted)
        self.loiter = next(z for z in zones if z.loiter_dwell_s)
        self.crowd = next(z for z in zones if z.crowd_threshold)
        self.zones = zones

    def edge_point(self) -> Point:
        r = self.rng.random()
        u = self.rng.uniform(0.05, 0.95)
        if r < 0.4:
            return (0.0, u * 0.6 + 0.35)
        if r < 0.8:
            return (1.0, u * 0.6 + 0.35)
        return (u * 0.6, 1.0)

    def inside(self, zone: Zone, margin: float = 0.03) -> Point:
        xs = [p[0] for p in zone.polygon]
        ys = [p[1] for p in zone.polygon]
        while True:
            p = (
                self.rng.uniform(min(xs) + margin, max(xs) - margin),
                self.rng.uniform(min(ys) + margin, max(ys) - margin),
            )
            if point_in_polygon(*p, zone.polygon):
                return p

    def walk(
        self, t: float, a: Point, b: Point, speed: float
    ) -> list[tuple[float, Point]]:
        n = max(1, int(math.dist(a, b) / speed / self.dt))
        bend = self.rng.uniform(-0.04, 0.04)
        out = []
        for i in range(1, n + 1):
            s = i / n
            off = bend * math.sin(math.pi * s)
            out.append(
                (
                    t + i * self.dt,
                    (a[0] + (b[0] - a[0]) * s + off, a[1] + (b[1] - a[1]) * s),
                )
            )
        return out

    def mill(
        self, t: float, anchor: Point, dur: float, radius: float, zone: Zone
    ) -> list[tuple[float, Point]]:
        out, p = [], anchor
        for i in range(1, int(dur / self.dt) + 1):
            q = (p[0] + self.rng.gauss(0, 0.004), p[1] + self.rng.gauss(0, 0.004))
            if math.dist(q, anchor) > radius or not point_in_polygon(*q, zone.polygon):
                q = (p[0] + (anchor[0] - p[0]) * 0.3, p[1] + (anchor[1] - p[1]) * 0.3)
            p = q
            out.append((t + i * self.dt, p))
        return out

    def hits_restricted(self, path: Sequence[tuple[float, Point]]) -> bool:
        return any(point_in_polygon(*p, self.restricted.polygon) for _, p in path)

    def safe_walk(
        self,
        t: float,
        a: Point | None,
        b: Point | None,
        speed: float,
        allow_restricted: bool = False,
    ) -> list[tuple[float, Point]]:
        """Camino desde/hacia un borde del frame (None) que no cruza la zona restringida."""
        path: list[tuple[float, Point]] = []
        for _ in range(50):
            path = self.walk(t, a or self.edge_point(), b or self.edge_point(), speed)
            if allow_restricted or not self.hits_restricted(path):
                return path
        return path


def _transit(sc: _Scene, pid: int, t0: float, speed: tuple[float, float]) -> Person:
    for _ in range(50):
        path = sc.walk(t0, sc.edge_point(), sc.edge_point(), sc.rng.uniform(*speed))
        if len(path) > 10 and not sc.hits_restricted(path):
            return Person(pid, "transit", [(t0, path[0][1]), *path])
    return Person(pid, "transit", [])


def _visit(
    sc: _Scene,
    pid: int,
    t0: float,
    zone: Zone,
    stay: float,
    radius: float,
    behavior: str,
) -> Person:
    anchor = sc.inside(zone)
    speed = sc.rng.uniform(0.03, 0.05)
    # Solo la propia visita a una zona restringida puede cruzarla.
    free = zone.restricted
    go = sc.safe_walk(t0, None, anchor, speed, allow_restricted=free)
    t_arrive = go[-1][0]
    stay_path = sc.mill(t_arrive, anchor, stay, radius, zone)
    end_t, end_p = stay_path[-1] if stay_path else (t_arrive, anchor)
    back = sc.safe_walk(end_t, end_p, None, speed, allow_restricted=free)
    return Person(pid, behavior, [*go, *stay_path, *back])


def _first_inside(path: Sequence[tuple[float, Point]], zone: Zone) -> float | None:
    return next((t for t, p in path if point_in_polygon(*p, zone.polygon)), None)


def _crowd_truth(
    people: Sequence[Person], zone: Zone, dt: float, duration: float
) -> list[tuple[float, float]]:
    assert zone.crowd_threshold is not None
    pos: dict[int, dict[int, Point]] = {}
    for p in people:
        for t, xy in p.path:
            pos.setdefault(round(t / dt), {})[p.pid] = xy
    spans, since = [], None
    for k in range(int(duration / dt) + 1):
        n = sum(point_in_polygon(*xy, zone.polygon) for xy in pos.get(k, {}).values())
        if n >= zone.crowd_threshold:
            since = k * dt if since is None else since
        elif since is not None:
            if k * dt - since >= zone.crowd_dwell_s:
                spans.append((since + zone.crowd_dwell_s, k * dt))
            since = None
    return spans


def generate(
    zones: Sequence[Zone],
    seed: int,
    perturb: Perturbation = TRAIN,
    duration_s: float = 180.0,
    dt: float = 0.2,
) -> Scenario:
    rng = random.Random(seed)
    sc = _Scene(zones, rng, dt)
    lz, rz, cz = sc.loiter, sc.restricted, sc.crowd
    assert lz.loiter_dwell_s is not None and cz.crowd_threshold is not None
    T = lz.loiter_dwell_s
    people: list[Person] = []

    def start(max_len: float) -> float:
        return rng.uniform(0.0, max(1.0, duration_s - max_len))

    for _ in range(rng.randint(3, 7)):
        people.append(
            _transit(sc, len(people), rng.uniform(0, duration_s - 20), (0.03, 0.07))
        )
    if rng.random() < 0.5:  # tránsito lento (hard negative de merodeo)
        a = (min(p[0] for p in lz.polygon) - 0.02, sc.inside(lz)[1])
        b = (
            max(p[0] for p in lz.polygon) + 0.02,
            min(0.88, a[1] + rng.uniform(-0.1, 0.1)),
        )
        t0 = start(80)
        path = sc.walk(t0, a, b, rng.uniform(0.006, 0.009))
        people.append(Person(len(people), "slow_transit", [(t0, a), *path]))
    if rng.random() < 0.5:  # merodeo
        stay = rng.uniform(1.3 * T, 2.5 * T)
        p = _visit(sc, len(people), start(stay + 40), lz, stay, 0.03, "loiter")
        entered = _first_inside(p.path, lz)
        if entered is not None:
            last_in = max(t for t, xy in p.path if point_in_polygon(*xy, lz.polygon))
            if last_in - entered >= T + 2.0:
                p.loiter = (entered + T, last_in)
        people.append(p)
    if rng.random() < 0.4:  # espera corta (hard negative)
        people.append(
            _visit(
                sc,
                len(people),
                start(T + 40),
                lz,
                rng.uniform(0.3, 0.7) * T,
                0.02,
                "short_wait",
            )
        )
    if rng.random() < 0.5:  # intrusión
        people.append(
            _visit(
                sc,
                len(people),
                start(50),
                rz,
                rng.uniform(1.5, 10.0),
                0.02,
                "intrusion",
            )
        )
    if rng.random() < 0.35:  # trabajador autorizado
        zone = rz if rng.random() < 0.5 else lz
        stay = rng.uniform(1.5 * T, 3.0 * T)
        p = _visit(
            sc, len(people), start(stay + 40), zone, stay, 0.04, "authorized_worker"
        )
        p.authorized = True
        if zone is lz:
            entered = _first_inside(p.path, lz)
            last_w = max(
                (t for t, xy in p.path if point_in_polygon(*xy, lz.polygon)),
                default=None,
            )
            if (
                entered is not None
                and last_w is not None
                and last_w - entered >= T + 2.0
            ):
                p.loiter = (entered + T, last_w)
        people.append(p)
    if (
        rng.random() < 0.4
    ):  # parado junto al borde de la zona restringida (hard negative)
        xs = [q[0] for q in rz.polygon]
        ys = [q[1] for q in rz.polygon]
        anchor = (
            min(xs) - rng.uniform(0.012, 0.025),
            rng.uniform(min(ys) + 0.05, max(ys) - 0.05),
        )
        t0 = start(60)
        go = sc.safe_walk(t0, None, anchor, 0.04)
        hold = [
            (go[-1][0] + (i + 1) * dt, anchor)
            for i in range(int(rng.uniform(15, 40) / dt))
        ]
        back = sc.safe_walk(hold[-1][0], anchor, None, 0.04)
        people.append(Person(len(people), "border_stand", [*go, *hold, *back]))
    if rng.random() < 0.4:  # grupo que se reúne (umbral o umbral-1 personas)
        k = cz.crowd_threshold - (1 if rng.random() < 0.4 else 0)
        anchor = sc.inside(cz, margin=0.08)
        t0 = start(70)
        gos = []
        for _ in range(k):
            spot = (
                anchor[0] + rng.uniform(-0.04, 0.04),
                anchor[1] + rng.uniform(-0.04, 0.04),
            )
            gos.append(
                (
                    spot,
                    sc.safe_walk(
                        t0 + rng.uniform(0, 4), None, spot, rng.uniform(0.035, 0.05)
                    ),
                )
            )
        # Todos se quedan hasta el mismo instante; permanencia muy por debajo del umbral de merodeo.
        t_end = max(go[-1][0] for _, go in gos) + rng.uniform(8.0, 0.4 * T)
        for spot, go in gos:
            hold = sc.mill(go[-1][0], spot, t_end - go[-1][0], 0.02, cz)
            end_t, end_p = hold[-1] if hold else go[-1]
            back = sc.safe_walk(end_t, end_p, None, 0.045)
            people.append(Person(len(people), "group", [*go, *hold, *back]))

    obs, track_ids, info, shakes = _observe(people, perturb, rng, dt, duration_s, sc)
    stream = zones[0].stream_id
    gt: list[GroundTruthEvent] = []
    for p in people:
        tids = tuple(track_ids[p.pid])
        if p.loiter:
            gt.append(
                GroundTruthEvent(
                    EventType.LOITERING, stream, lz.name, *p.loiter, tids, p.authorized
                )
            )
        t_in = _first_inside(p.path, rz)
        if t_in is not None:
            last_in = max(t for t, xy in p.path if point_in_polygon(*xy, rz.polygon))
            gt.append(
                GroundTruthEvent(
                    EventType.INTRUSION,
                    stream,
                    rz.name,
                    t_in,
                    last_in,
                    tids,
                    p.authorized,
                )
            )
    for s, e in _crowd_truth(people, cz, dt, duration_s):
        gt.append(GroundTruthEvent(EventType.CROWDING, stream, cz.name, s, e))
    behaviors: dict[str, int] = {}
    for p in people:
        behaviors[p.behavior] = behaviors.get(p.behavior, 0) + 1
    return Scenario(seed, stream, duration_s, obs, gt, behaviors, info, shakes)


def _observe(
    people: Sequence[Person],
    pr: Perturbation,
    rng: random.Random,
    dt: float,
    duration: float,
    sc: _Scene,
) -> tuple[
    list[TrackObservation],
    dict[int, list[int]],
    dict[int, dict],
    list[tuple[float, float]],
]:
    shakes: list[tuple[float, float]] = []
    t = 0.0
    while t < duration:
        if rng.random() < pr.shake_prob:
            shakes.append((t, t + rng.uniform(1.0, 4.0)))
        t += 10.0
    stream = sc.zones[0].stream_id
    next_tid = 1
    obs: list[TrackObservation] = []
    track_ids: dict[int, list[int]] = {}
    info: dict[int, dict] = {}
    for p in people:
        tid = next_tid
        next_tid += 1
        track_ids[p.pid] = [tid]
        gaps = []
        if p.path and rng.random() < pr.occlusion_prob:
            g0 = rng.uniform(p.path[0][0], p.path[-1][0])
            gaps.append((g0, g0 + rng.uniform(*pr.occlusion_s)))
        switch_t: float | None = (
            rng.uniform(p.path[0][0], p.path[-1][0])
            if p.path and rng.random() < pr.id_switch_prob
            else None
        )
        meta = {
            "behavior": p.behavior,
            "authorized": p.authorized,
            "occlusions": gaps,
            "id_switch_t": switch_t,
        }
        info[tid] = meta
        for t, (x, y) in p.path:
            if (
                t > duration
                or rng.random() < pr.dropout
                or any(a <= t <= b for a, b in gaps)
            ):
                continue
            if switch_t is not None and t >= switch_t:
                tid, switch_t = next_tid, None
                next_tid += 1
                track_ids[p.pid].append(tid)
                info[tid] = meta
            sx = sy = 0.0
            if any(a <= t <= b for a, b in shakes):
                sx = pr.shake_amp * math.sin(t * 9.0)
                sy = pr.shake_amp * math.cos(t * 7.0)
            obs.append(
                TrackObservation(
                    stream,
                    tid,
                    round(t, 3),
                    x + rng.gauss(0, pr.noise) + sx,
                    y + rng.gauss(0, pr.noise) + sy,
                    p.authorized,
                )
            )
    obs.sort(key=lambda o: (o.t, o.track_id))
    return obs, track_ids, info, shakes
