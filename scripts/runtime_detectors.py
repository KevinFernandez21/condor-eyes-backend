"""Issue #39: detectores standalone frente a detectores por el runtime multiagente.

Ejecuta los mismos detectores (armas v3 #1/#25 y vigilancia #9) sobre el mismo
video grabado en tres modos, cada uno en un proceso nuevo:

- `standalone`: OpenCV lee el video y se llama a `Detector.infer()` directamente.
- `lockstep`: `LiveVideoPipeline` → `DetectorProcessor` → `MetadataPublisher` →
  `InMemoryHub` → `AgentRuntime` (siete roles). La fuente lee el frame siguiente
  solo cuando el anterior llegó al bus, así que no se descarta ninguno y se puede
  comparar frame a frame.
- `realtime`: igual, pero la fuente emite al ritmo del video (30 FPS), como una
  cámara. Mide latencia, frames descartados y sobrecarga en condiciones reales.

uv run python scripts/runtime_detectors.py all --video data/raw/cctv-weapon/evaluation.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GB = 2**30


def build_detectors(which: list[str]) -> list[Any]:
    """Una instancia por modelo; la configuración por defecto de cada uno."""
    dets: list[Any] = []
    if "firearm" in which:
        from firearm import FirearmDetector, load_config

        dets.append(FirearmDetector(load_config("configs/firearm.toml")))
    if "surveillance" in which:
        from surveillance import SurveillanceDetector, load_surveillance_config

        dets.append(
            SurveillanceDetector(load_surveillance_config("configs/surveillance.toml"))
        )
    return dets


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def _peak_mem() -> dict[str, float]:
    import psutil  # type: ignore[import-untyped]

    m = psutil.Process().memory_info()
    return {
        "peak_working_set_gb": getattr(m, "peak_wset", m.rss) / GB,
        "peak_private_gb": getattr(m, "peak_pagefile", getattr(m, "private", 0)) / GB,
    }


def run_standalone(video: str, which: list[str], max_frames: int) -> dict[str, Any]:
    import cv2

    from pipeline.detection import DetectorProcessor
    from pipeline.sources import Frame

    proc = DetectorProcessor(build_detectors(which))
    cap = cv2.VideoCapture(video)
    _, first = cap.read()
    proc.warmup(first)  # mismo warmup previo que en el runtime
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames: list[dict[str, Any]] = []
    while len(frames) < max_frames:
        t_read = time.perf_counter()
        ok, img = cap.read()
        if not ok:
            break
        dets = proc("cam-1", Frame(width=img.shape[1], height=img.shape[0], data=img))
        frames.append(
            {
                "index": len(frames),
                "detections": dets,
                "e2e_ms": (time.perf_counter() - t_read) * 1000,
            }
        )
    cap.release()
    proc.close()
    return {"frames": frames, "infer_ms": proc.infer_ms}


class VideoFileSource:
    """`FrameSource` sobre un archivo de video, solo para evaluar (el pipeline no acepta archivos)."""

    def __init__(
        self,
        video: str,
        max_frames: int,
        lockstep: threading.Event | None,
        fps: float | None,
    ) -> None:
        self.video, self.max_frames = video, max_frames
        self.lockstep, self.fps = lockstep, fps
        self.reads: dict[
            int, tuple[int, float]
        ] = {}  # id(array) → (índice en el video, t_lectura)
        self.index = 0
        self._cap: Any = None
        self._next_t = 0.0
        self.exhausted = threading.Event()

    def open(self) -> None:
        import cv2

        from pipeline.sources import SourceError

        if (
            self.exhausted.is_set()
        ):  # el pipeline reintenta al acabar: no reabrir el archivo
            raise SourceError("video ya consumido")
        self._cap = cv2.VideoCapture(self.video)

    def read(self) -> Any:
        from pipeline.sources import Frame, SourceError

        if self.index >= self.max_frames:
            self.exhausted.set()
            time.sleep(0.05)
            raise SourceError("fin del video de prueba")
        if self.lockstep is not None:
            self.lockstep.wait()
            self.lockstep.clear()
        elif self.fps:
            now = time.perf_counter()
            if self._next_t and now < self._next_t:
                time.sleep(self._next_t - now)
            self._next_t = max(now, self._next_t) + 1.0 / self.fps
        t = time.perf_counter()
        ok, img = self._cap.read()
        if not ok:
            self.exhausted.set()
            raise SourceError("fin del video")
        self.reads[id(img)] = (self.index, t)
        self.index += 1
        return Frame(width=img.shape[1], height=img.shape[0], data=img)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def run_runtime(
    video: str, which: list[str], max_frames: int, mode: str
) -> dict[str, Any]:
    from agents.handlers import build_default_handlers
    from agents.runtime import AgentRuntime
    from bus.hub import Topic
    from bus.memory import InMemoryHub
    from pipeline.detection import DetectorProcessor
    from pipeline.live_pipeline import LiveVideoPipeline, PipelineConfig
    from pipeline.publisher import MetadataPublisher
    from pipeline.shared_pipeline import StreamSource

    class Sink:
        def __init__(self) -> None:
            self.count = 0

        async def save(self, envelope: Any) -> None:
            self.count += 1

        async def send(self, envelope: Any) -> None:
            self.count += 1

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    def call(coro: Any, timeout: float = 30.0) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)

    async def make_hub() -> Any:
        return InMemoryHub()

    hub = call(make_hub())
    sink = Sink()
    runtime = AgentRuntime(
        hub, build_default_handlers(storage_sink=sink, alert_sink=sink)
    )
    received: list[tuple[int, float, dict[str, Any]]] = []

    async def collect() -> None:
        sub = hub.subscribe(Topic.DETECTIONS)
        async for env in sub:
            received.append(
                (
                    int(env.payload["frame_index"]),
                    time.perf_counter(),
                    dict(env.payload),
                )
            )

    collector = asyncio.run_coroutine_threadsafe(collect(), loop)
    call(runtime.start())

    gate = threading.Event() if mode == "lockstep" else None
    if gate is not None:
        gate.set()
    src = VideoFileSource(video, max_frames, gate, 30.0 if mode == "realtime" else None)
    proc = DetectorProcessor(build_detectors(which))
    import cv2

    cap = cv2.VideoCapture(video)
    _, first = cap.read()
    cap.release()
    proc.warmup(
        first
    )  # el engine se carga antes de abrir la cámara, como en producción
    # Mapa índice del pipeline → (índice en el video, t_lectura): el pipeline renumera
    # los frames que entrega y en tiempo real puede descartar algunos.
    origin: dict[int, tuple[int, float]] = {}
    current: dict[str, tuple[int, float]] = {}

    def processor(stream_id: str, frame: Any) -> list[dict[str, Any]]:
        current["src"] = src.reads.pop(id(frame.data), (-1, time.perf_counter()))
        return proc(stream_id, frame)

    def on_metadata(meta: Any) -> None:
        origin[meta.frame_index] = current["src"]
        if gate is not None:
            gate.set()

    publisher = MetadataPublisher(hub, loop)
    pipe = LiveVideoPipeline(
        lambda _s: src,
        config=PipelineConfig(max_buffered_frames=4, watchdog_timeout=60.0),
        processor=processor,
        publisher=publisher,
        on_metadata=on_metadata,
    )
    pipe.add_source(
        StreamSource("cam-1", "usb:0")
    )  # la URI no se usa: la fábrica devuelve el archivo
    pipe.start()
    deadline = time.time() + 900
    while not src.exhausted.is_set() and time.time() < deadline:
        time.sleep(0.05)
    t_end = time.time() + 60

    def busy() -> bool:
        depth = sum(st["queue_depth"] for st in pipe.health()["streams"].values())
        return bool(depth or publisher.pending or len(received) < len(origin))

    while busy() and time.time() < t_end:
        time.sleep(0.05)
    health = pipe.health()
    pipe.stop()
    call(runtime.wait_idle(timeout=10.0))
    agents = runtime.health()
    call(runtime.stop())
    collector.cancel()
    call(hub.close())
    proc.close()
    loop.call_soon_threadsafe(loop.stop)

    frames = []
    for pidx, t_recv, payload in sorted(received):
        vidx, t_read = origin.get(pidx, (-1, t_recv))
        frames.append(
            {
                "index": vidx,
                "pipeline_index": pidx,
                "detections": payload["detections"],
                "e2e_ms": (t_recv - t_read) * 1000,
            }
        )
    return {
        "frames": frames,
        "infer_ms": proc.infer_ms,
        "source_frames_read": src.index,
        "pipeline_health": health,
        "publisher_dropped": publisher.dropped,
        "agents": {
            k: {"processed": v["processed"], "failures": v["failures"]}
            for k, v in agents.items()
        },
        "sink_messages": sink.count,
    }


def run_mode(a: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    which = a.detectors.split(",")
    res = (
        run_standalone(a.video, which, a.max_frames)
        if a.mode == "standalone"
        else run_runtime(a.video, which, a.max_frames, a.mode)
    )
    e2e = [f["e2e_ms"] for f in res["frames"]]
    inf = res["infer_ms"]
    res["summary"] = {
        "mode": a.mode,
        "video": a.video,
        "detectors": which,
        "frames_out": len(res["frames"]),
        "wall_s": time.perf_counter() - t0,
        "infer_ms_p50": _pct(inf, 0.5),
        "infer_ms_p90": _pct(inf, 0.9),
        "e2e_ms_p50": _pct(e2e, 0.5),
        "e2e_ms_p90": _pct(e2e, 0.9),
        **_peak_mem(),
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(
        json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(res["summary"], ensure_ascii=False))


def _same(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> tuple[bool, float]:
    """Mismas detecciones (modelo, clase) y máxima diferencia de conf/caja."""
    if len(a) != len(b):
        return False, float("inf")
    key = lambda d: (d["model"], d["cls"], d["xyxy"][0], d["xyxy"][1])
    worst = 0.0
    for x, y in zip(sorted(a, key=key), sorted(b, key=key), strict=True):
        if (x["model"], x["cls"]) != (y["model"], y["cls"]):
            return False, float("inf")
        worst = max(
            worst,
            abs(x["conf"] - y["conf"]),
            *(abs(p - q) for p, q in zip(x["xyxy"], y["xyxy"], strict=True)),
        )
    return worst <= 1.0, worst


def _non_inference(res: dict[str, Any]) -> dict[str, float]:
    inf = res["infer_ms"]
    vals = []
    for f in res["frames"]:
        i = f.get("pipeline_index", f["index"])
        if 0 <= i < len(inf):
            vals.append(f["e2e_ms"] - inf[i])
    return {"p50": round(_pct(vals, 0.5), 2), "p90": round(_pct(vals, 0.9), 2)}


def compare(out_dir: Path) -> dict[str, Any]:
    load = lambda m: json.loads((out_dir / f"{m}.json").read_text(encoding="utf-8"))
    std, lock, rt = load("standalone"), load("lockstep"), load("realtime")
    by_idx = {f["index"]: f["detections"] for f in std["frames"]}
    same = 0
    diffs = []
    worst = 0.0
    for f in lock["frames"]:
        ok, w = _same(by_idx.get(f["index"], []), f["detections"])
        same += ok
        worst = max(worst, w if w != float("inf") else 0.0)
        if not ok:
            diffs.append(
                {
                    "index": f["index"],
                    "standalone": len(by_idx.get(f["index"], [])),
                    "runtime": len(f["detections"]),
                }
            )
    # En tiempo real se comparan solo los frames que el pipeline entregó.
    rt_same = sum(
        _same(by_idx.get(f["index"], []), f["detections"])[0] for f in rt["frames"]
    )

    def per_class(frames: list[dict[str, Any]]) -> dict[str, int]:
        c: dict[str, int] = {}
        for f in frames:
            for d in f["detections"]:
                k = f"{d['model']}:{d.get('label', d['cls'])}"
                c[k] = c.get(k, 0) + 1
        return dict(sorted(c.items()))

    s, l, r = std["summary"], lock["summary"], rt["summary"]
    return {
        "frames": {
            "standalone": s["frames_out"],
            "lockstep": l["frames_out"],
            "realtime": r["frames_out"],
            "realtime_source_read": rt["source_frames_read"],
        },
        "lockstep_identical_frames": same,
        "lockstep_max_abs_diff": worst,
        "lockstep_diffs": diffs[:10],
        "realtime_identical_frames": rt_same,
        "detections_per_class": {
            "standalone": per_class(std["frames"]),
            "lockstep": per_class(lock["frames"]),
        },
        "latency_ms": {
            m: {k: round(v, 2) for k, v in x.items() if k.endswith(("p50", "p90"))}
            for m, x in (("standalone", s), ("lockstep", l), ("realtime", r))
        },
        # Sobrecarga = e2e − inferencia, frame a frame (aísla el camino pipeline → bus → agente
        # de la variación de la propia inferencia entre procesos).
        "non_inference_ms": {
            m: _non_inference(x)
            for m, x in (("standalone", std), ("lockstep", lock), ("realtime", rt))
        },
        "memory_gb": {
            m: {k: round(v, 3) for k, v in x.items() if k.startswith("peak")}
            for m, x in (("standalone", s), ("lockstep", l), ("realtime", r))
        },
        "realtime": {
            "pipeline": rt["pipeline_health"],
            "publisher_dropped": rt["publisher_dropped"],
            "agents": rt["agents"],
        },
        "lockstep_agents": lock["agents"],
    }


def hardware() -> dict[str, Any]:
    import psutil  # type: ignore[import-untyped]
    import torch

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor(),
        "cpu_threads": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / GB, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument(
        "--mode", choices=["standalone", "lockstep", "realtime"], required=True
    )
    r.add_argument("--video", required=True)
    r.add_argument("--detectors", default="firearm,surveillance")
    r.add_argument("--max-frames", type=int, default=10_000)
    r.add_argument("--output", required=True)
    a_ = sub.add_parser("all")
    a_.add_argument("--video", required=True)
    a_.add_argument("--detectors", default="firearm,surveillance")
    a_.add_argument("--max-frames", type=int, default=10_000)
    a_.add_argument("--out-dir", default="reports/runtime")
    a = p.parse_args()
    if a.cmd == "run":
        run_mode(a)
        return
    out = Path(a.out_dir) / Path(a.video).stem
    for mode in ("standalone", "lockstep", "realtime"):
        subprocess.run(
            [
                sys.executable,
                __file__,
                "run",
                "--mode",
                mode,
                "--video",
                a.video,
                "--detectors",
                a.detectors,
                "--max-frames",
                str(a.max_frames),
                "--output",
                str(out / f"{mode}.json"),
            ],
            check=True,
        )
    report = {
        "hardware": hardware(),
        "video": a.video,
        "detectors": a.detectors,
        "comparison": compare(out),
    }
    (out / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["comparison"], indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    main()
