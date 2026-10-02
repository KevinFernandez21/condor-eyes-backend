"""Descriptores de apariencia para re-ID.

- `ColorEmbedder`: baseline no biométrico; histograma HSV de la mitad superior e
  inferior del recorte (color de la ropa), sin red neuronal.
- `CNNEmbedder`: backbone de torchvision (ImageNet) con o sin fine-tuning re-ID.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

H, W = 256, 128
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def read_crops(paths: Sequence[Path | str]) -> np.ndarray:
    """Recortes BGR → uint8 (N, H, W, 3) en RGB a 256×128."""
    import cv2

    out = np.empty((len(paths), H, W, 3), dtype="uint8")
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Recorte ilegible: {p}")
        out[i] = cv2.cvtColor(cv2.resize(img, (W, H)), cv2.COLOR_BGR2RGB)
    return out


class ColorEmbedder:
    name = "color-hsv"

    def __init__(self, bins: tuple[int, int, int] = (8, 4, 4)) -> None:
        self.bins = bins

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        import cv2

        feats = []
        for rgb in crops:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            parts = []
            # Torso y piernas; se descarta el 15 % superior (cabeza y cara).
            for y0, y1 in ((int(0.15 * H), int(0.55 * H)), (int(0.55 * H), H)):
                h = cv2.calcHist(
                    [hsv[y0:y1]],
                    [0, 1, 2],
                    None,
                    list(self.bins),
                    [0, 180, 0, 256, 0, 256],
                )
                parts.append(np.sqrt(h.flatten() / max(h.sum(), 1.0)))  # Hellinger
            feats.append(np.concatenate(parts))
        f = np.asarray(feats, dtype="float32")
        return f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-9)


def _backbone(arch: str) -> tuple[Any, int]:
    import torch
    import torchvision  # type: ignore[import-untyped]

    if arch == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        dim = m.fc.in_features
        m.fc = torch.nn.Identity()
    elif arch == "mobilenet_v3_small":
        m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        dim = m.classifier[0].in_features
        m.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")
    return m, dim


def build_net(arch: str, num_classes: int = 0) -> Any:
    """Backbone + BNNeck. En inferencia el embedding es la salida del BN."""
    import torch

    class ReIDNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone, self.dim = _backbone(arch)
            self.bn = torch.nn.BatchNorm1d(self.dim)
            self.bn.bias.requires_grad_(False)
            self.head = (
                torch.nn.Linear(self.dim, num_classes, bias=False)
                if num_classes
                else None
            )

        def forward(self, x: Any) -> Any:
            f = self.backbone(x)
            b = self.bn(f)
            if self.training and self.head is not None:
                return f, self.head(b)
            return b

    return ReIDNet()


def to_tensor(crops: np.ndarray, device: str) -> Any:
    import torch

    x = torch.from_numpy(crops).to(device).float().div_(255.0)
    x = (x - torch.from_numpy(MEAN).to(device)) / torch.from_numpy(STD).to(device)
    return x.permute(0, 3, 1, 2).contiguous()


class CNNEmbedder:
    def __init__(
        self,
        arch: str,
        weights: str | Path | None = None,
        device: str = "cuda",
        half: bool = True,
    ) -> None:
        import torch

        self.arch = arch
        self.name = f"{arch}-{'reid' if weights else 'imagenet'}"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.half = half and self.device != "cpu"
        net = build_net(arch)
        if weights:
            state = torch.load(weights, map_location="cpu")
            state = {k: v for k, v in state.items() if not k.startswith("head.")}
            net.load_state_dict(state)
        net.eval().to(self.device)
        self.net = net.half() if self.half else net

    @property
    def params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def __call__(self, crops: np.ndarray, batch: int = 128) -> np.ndarray:
        import torch

        outs = []
        with torch.inference_mode():
            for i in range(0, len(crops), batch):
                x = to_tensor(crops[i : i + batch], self.device)
                x = x.half() if self.half else x
                f = self.net(x) + self.net(
                    torch.flip(x, dims=[3])
                )  # promedio con espejo
                outs.append(
                    torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy()
                )
        return np.concatenate(outs) if outs else np.zeros((0, 1), dtype="float32")


def train_reid(
    arch: str,
    crops: np.ndarray,
    pids: Sequence[int],
    out: str | Path,
    epochs: int = 40,
    p: int = 16,
    k: int = 4,
    lr: float = 3.5e-4,
    seed: int = 0,
    log: Any = print,
) -> Path:
    """Fine-tuning con entropía cruzada (label smoothing) + triplete batch-hard (P×K).

    Los recortes ya están en memoria (uint8); el aumento (espejo, recorte con
    padding, borrado aleatorio) se hace en GPU para no depender de workers.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = {pid: i for i, pid in enumerate(sorted(set(pids)))}
    y_all = np.asarray([labels[x] for x in pids])
    by_id: dict[int, np.ndarray] = {i: np.where(y_all == i)[0] for i in labels.values()}
    net = build_net(arch, len(labels)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    data = torch.from_numpy(crops)  # CPU uint8
    steps = len(crops) // (p * k)
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for _ in range(steps):
            ids = rng.choice(len(by_id), p, replace=False)
            idx = np.concatenate(
                [rng.choice(by_id[i], k, replace=len(by_id[i]) < k) for i in ids]
            )
            x = to_tensor(data[idx].numpy(), dev)
            if rng.random() < 0.5:
                x = torch.flip(x, dims=[3])
            x = F.pad(x, (10, 10, 10, 10), mode="reflect")
            dx, dy = rng.integers(0, 21, 2)
            x = x[:, :, dy : dy + H, dx : dx + W]
            # Borrado aleatorio de un rectángulo por imagen.
            for i in range(len(x)):
                if rng.random() < 0.5:
                    eh, ew = rng.integers(20, 100), rng.integers(10, 50)
                    ey, ex = rng.integers(0, H - eh), rng.integers(0, W - ew)
                    x[i, :, ey : ey + eh, ex : ex + ew] = 0
            y = torch.from_numpy(y_all[idx]).to(dev)
            feat, logits = net(x)
            ce = F.cross_entropy(logits, y, label_smoothing=0.1)
            d = torch.cdist(feat, feat)
            same = y[:, None] == y[None, :]
            hardest_pos = (d * same).max(1).values
            hardest_neg = (d + same * 1e6).min(1).values
            tri = F.relu(hardest_pos - hardest_neg + 0.3).mean()
            loss = ce + tri
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        log(f"época {ep + 1}/{epochs} pérdida {tot / steps:.3f}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out)
    return out
