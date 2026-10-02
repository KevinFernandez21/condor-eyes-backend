"""Aumentos de dominio para acercar las fotos de Open Images a una cámara CCTV.

- `degrade`: pérdida de resolución, compresión JPEG fuerte, desenfoque (gaussiano o
  de movimiento), poca luz con ruido y escala de grises.
- `make_composites` (*copy-paste*): recorta "persona + arma" de Open Images, la
  reduce al tamaño de alguien lejano y la pega con bordes difuminados sobre frames
  reales de CCTV (MOT16 de train y escenas de train de Simuletic), conservando sus
  etiquetas y las del fondo.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    out = img
    if rng.random() < 0.7:  # resolución efectiva baja
        s = rng.uniform(0.25, 0.6)
        small = cv2.resize(
            out, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA
        )
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        if rng.random() < 0.5:
            out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.8))
        else:
            k = rng.choice([5, 7, 9])
            ker = np.zeros((k, k), dtype="float32")
            ker[k // 2, :] = 1.0 / k
            m = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1.0)
            out = cv2.filter2D(out, -1, cv2.warpAffine(ker, m, (k, k)))
    if rng.random() < 0.4:  # poca luz + ruido de sensor
        g = rng.uniform(1.4, 2.4)
        lut = (np.linspace(0, 1, 256) ** g * 255 * rng.uniform(0.5, 0.85)).astype(
            "uint8"
        )
        out = cv2.LUT(out, lut)
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, rng.uniform(3, 9), out.shape
        )
        out = np.clip(out.astype("float32") + noise, 0, 255).astype("uint8")
    if rng.random() < 0.3:  # cámaras nocturnas / IR en escala de grises
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if rng.random() < 0.8:
        q = rng.randint(15, 50)
        _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is not None:
            out = dec
    return out


def _read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows = []
    for ln in path.read_text().splitlines():
        if ln.strip():
            c, x, y, w, h = ln.split()[:5]
            rows.append((int(c), float(x), float(y), float(w), float(h)))
    return rows


def _write(
    img: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    out: Path,
    split: str,
    stem: str,
) -> None:
    import cv2

    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    (out / "labels" / split / f"{stem}.txt").write_text(
        "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels),
        encoding="utf-8",
    )


def make_degraded(
    src: str | Path, split: str, out: str | Path, frac: float = 0.5, seed: int = 0
) -> int:
    """Copias degradadas (con las mismas etiquetas) de una fracción de las imágenes."""
    import cv2

    rng = random.Random(seed)
    src, out = Path(src), Path(out)
    paths = sorted(
        p for p in (src / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    chosen = rng.sample(paths, int(len(paths) * frac))
    for p in chosen:
        img = cv2.imread(str(p))
        if img is None:
            continue
        _write(
            degrade(img, rng),
            _read_labels(src / "labels" / split / f"{p.stem}.txt"),
            out,
            "train",
            f"deg_{p.stem}",
        )
    return len(chosen)


def _armed_people(
    src: Path, split: str
) -> list[
    tuple[
        Path, tuple[float, float, float, float], list[tuple[float, float, float, float]]
    ]
]:
    """(imagen, recorte persona+arma, armas dentro del recorte) en coordenadas normalizadas."""
    out = []
    for lab in sorted((src / "labels" / split).glob("*.txt")):
        rows = _read_labels(lab)
        people = [r for r in rows if r[0] == 0]
        weapons = [r for r in rows if r[0] == 1]
        for _, px, py, pw, ph in people:
            p = (px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2)
            near = []
            for _, wx, wy, ww, wh in weapons:
                wb = (wx - ww / 2, wy - wh / 2, wx + ww / 2, wy + wh / 2)
                ix = max(0.0, min(p[2], wb[2]) - max(p[0], wb[0]))
                iy = max(0.0, min(p[3], wb[3]) - max(p[1], wb[1]))
                if ix * iy > 0.3 * (wb[2] - wb[0]) * (wb[3] - wb[1]):
                    near.append(wb)
            if near and 0.15 < ph < 0.98:
                x1 = min([p[0], *[w[0] for w in near]])
                y1 = min([p[1], *[w[1] for w in near]])
                x2 = max([p[2], *[w[2] for w in near]])
                y2 = max([p[3], *[w[3] for w in near]])
                img = src / "images" / split / f"{lab.stem}.jpg"
                if img.exists():
                    out.append((img, (x1, y1, x2, y2), [*near, p]))
    return out


def make_composites(
    src: str | Path,
    backgrounds: list[tuple[str | Path, str]],
    out: str | Path,
    n: int = 2000,
    height_px: tuple[int, int] = (50, 220),
    seed: int = 0,
) -> dict[str, Any]:
    """Pega personas armadas de Open Images, pequeñas y con borde suave, sobre fondos CCTV."""
    import cv2

    rng = random.Random(seed)
    src, out = Path(src), Path(out)
    sources = _armed_people(src, "train")
    bgs = [
        (p, Path(root) / "labels" / split / f"{p.stem}.txt")
        for root, split in backgrounds
        for p in sorted((Path(root) / "images" / split).iterdir())
        if p.suffix.lower() in IMAGE_EXTS
    ]
    made = 0
    for k in range(n):
        img_p, crop, boxes = rng.choice(sources)
        bg_p, bg_lab = rng.choice(bgs)
        im, bg = cv2.imread(str(img_p)), cv2.imread(str(bg_p))
        if im is None or bg is None:
            continue
        ih, iw = im.shape[:2]
        bh, bw = bg.shape[:2]
        cx1, cy1, cx2, cy2 = (
            int(crop[0] * iw),
            int(crop[1] * ih),
            int(crop[2] * iw),
            int(crop[3] * ih),
        )
        patch = im[cy1:cy2, cx1:cx2]
        if patch.size == 0 or patch.shape[0] < 20:
            continue
        target_h = min(rng.randint(*height_px), int(0.8 * bh))
        s = target_h / patch.shape[0]
        pw, ph = max(4, int(patch.shape[1] * s)), max(4, int(patch.shape[0] * s))
        if pw >= bw - 2:
            continue
        patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
        x0 = rng.randint(0, bw - pw)
        y0 = rng.randint(int(0.25 * bh), max(int(0.25 * bh), bh - ph))
        mask = np.zeros((ph, pw), dtype="float32")
        border = max(2, int(0.12 * min(pw, ph)))
        mask[border:-border, border:-border] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), border / 2)[..., None]
        roi = bg[y0 : y0 + ph, x0 : x0 + pw].astype("float32")
        bg[y0 : y0 + ph, x0 : x0 + pw] = (mask * patch + (1 - mask) * roi).astype(
            "uint8"
        )
        # Etiquetas del fondo que el parche no tapa (más del 50 %).
        labels = []
        for c, x, y, w, h in _read_labels(bg_lab):
            fx1, fy1, fx2, fy2 = (
                (x - w / 2) * bw,
                (y - h / 2) * bh,
                (x + w / 2) * bw,
                (y + h / 2) * bh,
            )
            ix = max(0.0, min(fx2, x0 + pw) - max(fx1, x0))
            iy = max(0.0, min(fy2, y0 + ph) - max(fy1, y0))
            if ix * iy < 0.5 * (fx2 - fx1) * (fy2 - fy1):
                labels.append((c, x, y, w, h))
        for i, (bx1, by1, bx2, by2) in enumerate(boxes):
            # Coordenadas de Open Images (normalizadas) → recorte → fondo.
            gx1 = x0 + (bx1 * iw - cx1) * s
            gy1 = y0 + (by1 * ih - cy1) * s
            gx2 = x0 + (bx2 * iw - cx1) * s
            gy2 = y0 + (by2 * ih - cy1) * s
            cls = 0 if i == len(boxes) - 1 else 1
            labels.append(
                (
                    cls,
                    (gx1 + gx2) / 2 / bw,
                    (gy1 + gy2) / 2 / bh,
                    (gx2 - gx1) / bw,
                    (gy2 - gy1) / bh,
                )
            )
        if rng.random() < 0.5:
            bg = degrade(bg, rng)
        _write(bg, labels, out, "train", f"cp_{k:05d}")
        made += 1
    return {
        "composites": made,
        "armed_people_sources": len(sources),
        "backgrounds": len(bgs),
    }
