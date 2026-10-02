"""Modelo temporal aprendido para merodeo, para compararlo con la regla.

Es un MLP pequeño sobre las mismas `LoiterFeatures` que usa la regla. Se entrena
con torch y la inferencia se hace con numpy, para que el motor no cargue torch en
cada actualización. Las etiquetas salen del guion de los escenarios de train.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .engine import EventEngine, LoiterFeatures
from .model import EventPolicy, EventType, TrackObservation, Zone
from .simulate import Scenario


def feature_vector(f: LoiterFeatures, zone: Zone) -> list[float]:
    assert zone.loiter_dwell_s is not None
    T, d = zone.loiter_dwell_s, zone.loiter_max_disp
    return [
        f.dwell_s / T,
        f.window_s / T,
        f.net_disp / d,
        f.path_len / d,
        f.straightness,
    ]


class _Recorder:
    """Decisor que registra features y etiquetas y nunca dispara."""

    def __init__(self) -> None:
        self.xs: list[list[float]] = []
        self.ys: list[float] = []
        self.spans: list[tuple[float, float, set[int]]] = []
        self.obs: TrackObservation | None = None

    def __call__(self, f: LoiterFeatures, zone: Zone) -> tuple[bool, float]:
        obs = self.obs
        assert obs is not None
        self.xs.append(feature_vector(f, zone))
        hit = any(s <= obs.t <= e and obs.track_id in ids for s, e, ids in self.spans)
        self.ys.append(float(hit))
        return False, 0.0


def collect(
    scenarios: Sequence[Scenario], zones: Sequence[Zone], policy: EventPolicy
) -> tuple[np.ndarray, np.ndarray]:
    """Features por actualización dentro de zonas de merodeo y etiqueta del guion."""
    rec = _Recorder()
    for sc in scenarios:
        rec.spans = [
            (g.start_t, g.end_t, set(g.track_ids))
            for g in sc.ground_truth
            if g.type == EventType.LOITERING
        ]
        engine = EventEngine(zones, policy, loiter_decider=rec)
        for obs in sc.observations:
            rec.obs = obs
            engine.update(obs)
    return np.asarray(rec.xs, dtype="float32"), np.asarray(rec.ys, dtype="float32")


@dataclass
class LearnedLoiter:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    threshold: float = 0.5
    # Muestras consecutivas sobre el umbral antes de disparar (suavizado temporal).
    consecutive: int = 1
    _streak: dict[tuple[str, int, str], int] = field(default_factory=dict, repr=False)

    def reset(self) -> None:
        self._streak.clear()

    def prob(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        return 1.0 / (1.0 + np.exp(-(h @ self.w2 + self.b2)))

    def __call__(self, f: LoiterFeatures, zone: Zone) -> tuple[bool, float]:
        p = float(self.prob(np.asarray([feature_vector(f, zone)], dtype="float32"))[0])
        n = self._streak.get(f.key, 0) + 1 if p >= self.threshold else 0
        self._streak[f.key] = n
        return n >= self.consecutive, p


def train(
    x: np.ndarray, y: np.ndarray, epochs: int = 300, hidden: int = 16, seed: int = 0
) -> LearnedLoiter:
    import torch

    torch.manual_seed(seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    l1 = torch.nn.Linear(x.shape[1], hidden)
    l2 = torch.nn.Linear(hidden, 1)
    model = torch.nn.Sequential(l1, torch.nn.ReLU(), l2)
    pos = float(y.sum())
    weight = torch.tensor((len(y) - pos) / max(pos, 1.0))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt).squeeze(1), yt)
        loss.backward()
        opt.step()
    return LearnedLoiter(
        w1=l1.weight.detach().numpy().T.copy(),
        b1=l1.bias.detach().numpy().copy(),
        w2=l2.weight.detach().numpy()[0].copy(),
        b2=float(l2.bias.detach().numpy()[0]),
    )


def calibrate_events(
    model: LearnedLoiter,
    zones: Sequence[Zone],
    policy: EventPolicy,
    val: Sequence[Scenario],
    thresholds: Sequence[float] = (0.6, 0.7, 0.8, 0.9, 0.95),
    consecutive: Sequence[int] = (1, 5, 10, 25),
) -> dict[str, float]:
    """Elige umbral y suavizado por F1 de *evento* de merodeo en validación."""
    from .metrics import TypeStats, match_events

    best = {"f1": -1.0, "threshold": model.threshold, "consecutive": 1}
    for thr in thresholds:
        for k in consecutive:
            model.threshold, model.consecutive = thr, k
            stats: dict[EventType, TypeStats] = {}
            for sc in val:
                model.reset()
                preds = EventEngine(zones, policy, loiter_decider=model).process(
                    sc.observations
                )
                match_events(
                    [p for p in preds if p.type == EventType.LOITERING],
                    [g for g in sc.ground_truth if g.type == EventType.LOITERING],
                    stats,
                )
            st = stats.get(EventType.LOITERING, TypeStats())
            f1 = 2 * st.tp / (2 * st.tp + st.fp + st.fn) if st.tp else 0.0
            if f1 > best["f1"]:
                best = {"f1": f1, "threshold": thr, "consecutive": k}
    model.threshold, model.consecutive = best["threshold"], int(best["consecutive"])
    model.reset()
    return best
