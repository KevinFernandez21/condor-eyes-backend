"""Presupuesto de memoria y throughput en la laptop como proxy de la Jetson (issue #23).

Cada configuración corre en un proceso nuevo (`run_config`) con:

- N streams de video (un `VideoCapture` por cámara, 1080p o 720p) en el mismo proceso;
- uno o dos detectores YOLOv8n en OpenVINO FP16, **un solo modelo compilado por
  detector** compartido por todos los streams: los N frames de cada ciclo van en un lote;
- opcionalmente re-ID (ResNet18) sobre las personas detectadas y verificación facial
  (YuNet + SFace) sobre la zona de la cabeza, también en OpenVINO FP16.

Mide FPS por stream de extremo a extremo (decodificación incluida), latencia por
ciclo p50/p90, pico de memoria del proceso (working set y commit privado) y, desde
el orquestador, el pico de memoria de GPU compartida/dedicada del proceso
(contadores `GPU Process Memory` de Windows). Sin CUDA, TensorRT ni NVDEC.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

EDGE = Path("weights/edge")
VIDEOS_1080 = ["data/raw/mot/videos/MOT16-02.mp4", "data/raw/mot/videos/MOT16-09.mp4"]
GB = 2**30


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def hardware() -> dict[str, Any]:
    import openvino as ov  # type: ignore[import-untyped]
    import psutil  # type: ignore[import-untyped]

    core = ov.Core()
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "openvino": ov.__version__,
        "cpu": core.get_property("CPU", "FULL_DEVICE_NAME"),
        "cpu_threads": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / GB, 1),
        "devices": {
            d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices
        },
    }
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    return info


def make_720p(src: str) -> str:
    """Versión 720p del video (simula una cámara configurada a 720p)."""
    import cv2

    dst = src.replace(".mp4", "_720p.mp4")
    if Path(dst).exists():
        return dst
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = cv2.VideoWriter(dst, cv2.VideoWriter.fourcc(*"mp4v"), fps, (1280, 720))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        w.write(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA))
    w.release()
    cap.release()
    return dst


class _Streams:
    """N cámaras simuladas: cada una con su decodificador y un desfase inicial."""

    def __init__(self, n: int, source: str) -> None:
        import cv2

        vids = VIDEOS_1080 if source == "1080p" else [make_720p(v) for v in VIDEOS_1080]
        self.caps = []
        for i in range(n):
            cap = cv2.VideoCapture(vids[i % len(vids)])
            cap.set(cv2.CAP_PROP_POS_FRAMES, (i * 37) % 400)
            self.caps.append((cap, vids[i % len(vids)]))

    def read(self) -> list[Any]:
        import cv2

        frames = []
        for k, (cap, path) in enumerate(self.caps):
            ok, f = cap.read()
            if not ok:  # bucle infinito sobre el video
                cap.release()
                cap = cv2.VideoCapture(path)
                self.caps[k] = (cap, path)
                ok, f = cap.read()
            frames.append(f)
        return frames


def run_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta una configuración y devuelve sus métricas (llamar en un proceso nuevo)."""
    import cv2
    import numpy as np
    import psutil  # type: ignore[import-untyped]

    proc = psutil.Process()
    device = cfg["device"]
    ov_device = device.split(":", 1)[1].upper()  # intel:gpu.0 → GPU.0
    streams = _Streams(cfg["streams"], cfg["source"])
    rss_after_imports = proc.memory_info().rss

    if cfg.get("runtime", "lean") == "ultralytics":
        from ultralytics import YOLO

        uy = [
            (YOLO(str(EDGE / d["model"]), task="detect"), d["imgsz"])
            for d in cfg["detectors"]
        ]

        def make(m: Any, imgsz: int) -> Any:
            def run(frames: list[Any]) -> list[list[tuple[int, float, list[float]]]]:
                res = m.predict(
                    frames, imgsz=imgsz, device=device, verbose=False, conf=0.3
                )
                return [
                    list(
                        zip(
                            map(int, r.boxes.cls.tolist()),
                            r.boxes.conf.tolist(),
                            r.boxes.xyxy.tolist(),
                            strict=True,
                        )
                    )
                    for r in res
                ]

            return run

        detectors = [make(m, sz) for m, sz in uy]
    else:
        from .ovdetect import OVDetector

        detectors = [
            OVDetector(EDGE / d["model"], ov_device, d["imgsz"]).predict
            for d in cfg["detectors"]
        ]
    reid = face_det = face_rec = None
    if cfg.get("reid") or cfg.get("face"):
        import openvino as ov  # type: ignore[import-untyped]

        core = ov.Core()
        if cfg.get("reid"):
            reid = core.compile_model(str(EDGE / "reid_resnet18_fp16.xml"), ov_device)
        if cfg.get("face"):
            yunet = core.read_model(str(EDGE / "face_detection_yunet_2023mar_fp16.xml"))
            yunet.reshape(
                [1, 3, 160, 160]
            )  # entrada fija del export; OpenCV la redimensiona igual
            face_det = core.compile_model(yunet, ov_device)
            sface = core.read_model(
                str(EDGE / "face_recognition_sface_2021dec_fp16.xml")
            )
            sface.reshape(
                [-1, 3, 112, 112]
            )  # lote dinámico: todas las caras del ciclo juntas
            face_rec = core.compile_model(sface, ov_device)

    def cycle() -> dict[str, float]:
        t = {}
        s = time.perf_counter()
        frames = streams.read()
        t["decode"] = time.perf_counter() - s
        people: list[tuple[int, list[float]]] = []
        s = time.perf_counter()
        for k, det in enumerate(detectors):
            res = det(frames)
            if k == 0:
                for i, dets in enumerate(res):
                    boxes = [b for c, _, b in dets if c == 0]
                    people += [(i, b) for b in boxes[: cfg.get("reid_per_frame", 4)]]
        t["detect"] = time.perf_counter() - s
        s = time.perf_counter()
        if reid is not None and people:
            crops = []
            for i, (x1, y1, x2, y2) in people:
                c = frames[i][max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)]
                if c.size:
                    crops.append(
                        cv2.resize(c, (128, 256)).transpose(2, 0, 1).astype("float32")
                        / 255.0
                    )
            if crops:
                reid([np.stack(crops)])
        t["reid"] = time.perf_counter() - s
        s = time.perf_counter()
        if face_det is not None and face_rec is not None and people:
            heads = []
            for i, (x1, y1, x2, y2) in people[
                : cfg.get("faces_per_cycle", cfg["streams"])
            ]:
                h = frames[i][
                    max(0, int(y1)) : int(y1 + 0.3 * (y2 - y1)),
                    max(0, int(x1)) : int(x2),
                ]
                if h.size:
                    heads.append(h)
            for h in heads:  # YuNet tiene entrada fija; una cabeza por inferencia
                face_det(
                    [
                        cv2.resize(h, (160, 160))
                        .transpose(2, 0, 1)[None]
                        .astype("float32")
                    ]
                )
            if heads:
                face_rec(
                    [
                        np.stack(
                            [
                                cv2.resize(h, (112, 112))
                                .transpose(2, 0, 1)
                                .astype("float32")
                                for h in heads
                            ]
                        )
                    ]
                )
        t["face"] = time.perf_counter() - s
        return t

    for _ in range(cfg.get("warmup", 5)):
        cycle()
    psutil.cpu_percent(None)
    times: list[dict[str, float]] = []
    t0 = time.perf_counter()
    while len(times) < cfg.get("cycles", 60) and time.perf_counter() - t0 < cfg.get(
        "max_s", 40
    ):
        s = time.perf_counter()
        part = cycle()
        part["total"] = time.perf_counter() - s
        times.append(part)
    wall = time.perf_counter() - t0
    mem = proc.memory_info()
    total = [x["total"] * 1000 for x in times]
    n = cfg["streams"]
    return {
        **cfg,
        "cycles_measured": len(times),
        "fps_per_stream": len(times) / wall,
        "fps_total": len(times) * n / wall,
        "cycle_ms_p50": _pct(total, 0.5),
        "cycle_ms_p90": _pct(total, 0.9),
        "ms_mean": {k: 1000 * statistics.fmean(x[k] for x in times) for k in times[0]},
        "rss_after_imports_gb": rss_after_imports / GB,
        "peak_working_set_gb": getattr(mem, "peak_wset", mem.rss) / GB,
        "peak_private_gb": getattr(mem, "peak_pagefile", getattr(mem, "private", 0))
        / GB,
        "system_cpu_percent": psutil.cpu_percent(None),
    }


def _gpu_mem_gb(pids: list[int]) -> float:
    """Memoria de GPU (compartida + dedicada) de los procesos según los contadores de Windows."""
    paths = ",".join(
        f"'\\GPU Process Memory(pid_{pid}_*)\\{kind} Usage'"
        for pid in pids
        for kind in ("Shared", "Dedicated")
    )
    cmd = f"$s=(Get-Counter {paths} -ErrorAction SilentlyContinue).CounterSamples; ($s | Measure-Object CookedValue -Sum).Sum"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-c", cmd],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        ).stdout
        return float(out.strip() or 0) / GB
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def run_isolated(
    cfg: dict[str, Any], script: str = "scripts/edge_budget.py"
) -> dict[str, Any]:
    """Lanza `run_config` en un proceso Python nuevo y muestrea su memoria de GPU."""
    p = subprocess.Popen(
        [sys.executable, script, "run", "--config", json.dumps(cfg)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    peak = [0.0]
    stop = threading.Event()

    def sample() -> None:
        import psutil  # type: ignore[import-untyped]

        while not stop.is_set():
            # El python.exe del venv es un lanzador: el trabajo real ocurre en un hijo.
            try:
                pids = [p.pid] + [
                    c.pid for c in psutil.Process(p.pid).children(recursive=True)
                ]
            except psutil.NoSuchProcess:
                break
            peak[0] = max(peak[0], _gpu_mem_gb(pids))
            stop.wait(1.0)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    out, _ = p.communicate()
    stop.set()
    th.join(timeout=25)
    lines = [ln for ln in out.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        return {**cfg, "error": f"código {p.returncode}"}
    res = json.loads(lines[-1])
    res["peak_gpu_process_gb"] = peak[0]
    # Memoria total atribuible al proceso: RAM privada + memoria de GPU (en la iGPU
    # sale de la misma RAM del sistema, como en la memoria unificada de la Jetson).
    res["total_process_gb"] = res["peak_private_gb"] + peak[0]
    return res
