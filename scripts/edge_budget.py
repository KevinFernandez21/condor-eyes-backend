"""Presupuesto de memoria en la laptop como proxy de la Jetson (issue #23).

uv run python scripts/edge_budget.py matrix --output reports/edge/budget.json
uv run python scripts/edge_budget.py run --config '{"device": "intel:gpu.0", ...}'
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge.budget import hardware, run_config, run_isolated

COCO = {"640": "coco_640_fp16_openvino_model", "1280": "coco_1280_fp16_openvino_model"}
FIREARM = {"model": "firearm_v3_640_fp16_openvino_model", "imgsz": 640}


def matrix() -> list[dict]:
    cfgs: list[dict] = []
    for device in ("intel:gpu.0", "intel:cpu"):
        for imgsz in ("640", "1280"):
            for n in (1, 4, 8):
                base = {"device": device, "streams": n, "source": "1080p"}
                # Un solo detector: coste de un modelo unificado (misma arquitectura YOLOv8n).
                cfgs.append(
                    {
                        **base,
                        "name": f"unificado-{imgsz}",
                        "detectors": [{"model": COCO[imgsz], "imgsz": int(imgsz)}],
                    }
                )
                # Dos detectores: vigilancia (#9) + armas (#1), lo que hay hoy en main.
                cfgs.append(
                    {
                        **base,
                        "name": f"separados-{imgsz}+armas640",
                        "detectors": [
                            {"model": COCO[imgsz], "imgsz": int(imgsz)},
                            FIREARM,
                        ],
                    }
                )
        # Fallback: cámaras a 720p.
        for name, dets in (
            ("separados-640+armas640", [{"model": COCO["640"], "imgsz": 640}, FIREARM]),
            ("unificado-640", [{"model": COCO["640"], "imgsz": 640}]),
        ):
            cfgs.append(
                {
                    "device": device,
                    "streams": 8,
                    "source": "720p",
                    "name": name,
                    "detectors": dets,
                }
            )
        # Stack completo: detector(es) + re-ID + verificación facial.
        for n in (4, 8):
            for name, dets in (
                (
                    "stack-separados-640",
                    [{"model": COCO["640"], "imgsz": 640}, FIREARM],
                ),
                ("stack-unificado-640", [{"model": COCO["640"], "imgsz": 640}]),
            ):
                cfgs.append(
                    {
                        "device": device,
                        "streams": n,
                        "source": "1080p",
                        "name": name,
                        "detectors": dets,
                        "reid": True,
                        "face": True,
                        "reid_per_frame": 4,
                    }
                )
    return cfgs


def export(a: argparse.Namespace) -> None:
    """Exporta a OpenVINO FP16 todos los modelos del presupuesto (en `weights/edge/`)."""
    import shutil

    import openvino as ov  # type: ignore[import-untyped]
    import torch
    from ultralytics import YOLO

    from reid.embed import build_net

    out = Path("weights/edge")
    out.mkdir(parents=True, exist_ok=True)
    for name, pt, sz in (
        ("coco_640", a.coco, 640),
        ("coco_1280", a.coco, 1280),
        ("firearm_v3_640", a.firearm, 640),
    ):
        src = YOLO(pt).export(
            format="openvino", half=True, imgsz=sz, dynamic=True, batch=8
        )
        dst = out / f"{name}_fp16_openvino_model"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(src, dst)
    net = build_net("resnet18")
    state = torch.load(a.reid, map_location="cpu")
    net.load_state_dict({k: v for k, v in state.items() if not k.startswith("head.")})
    net.eval()
    onnx_path = out / "reid_resnet18.onnx"
    torch.onnx.export(
        net,
        torch.zeros(1, 3, 256, 128),
        onnx_path,
        input_names=["x"],
        output_names=["emb"],
        dynamic_axes={"x": {0: "n"}, "emb": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    ov.save_model(
        ov.convert_model(onnx_path),
        out / "reid_resnet18_fp16.xml",
        compress_to_fp16=True,
    )
    for onnx_name in ("face_detection_yunet_2023mar", "face_recognition_sface_2021dec"):
        model = ov.convert_model(str(Path(a.faceid) / f"{onnx_name}.onnx"))
        ov.save_model(model, out / f"{onnx_name}_fp16.xml", compress_to_fp16=True)
    print(f"Modelos OpenVINO FP16 en {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    e = sub.add_parser("export", help="Exporta los modelos a OpenVINO FP16")
    e.add_argument("--coco", default="weights/yolov8n.pt")
    e.add_argument(
        "--firearm", default="runs/firearm/yolov8n_firearm_v3/weights/best.pt"
    )
    e.add_argument("--reid", default="weights/reid/resnet18_market.pt")
    e.add_argument(
        "--faceid",
        default="weights/faceid",
        help="Carpeta con los ONNX de YuNet y SFace",
    )
    m = sub.add_parser("matrix")
    m.add_argument("--output", default="reports/edge/budget.json")
    m.add_argument(
        "--only", help="Filtra configuraciones cuyo nombre contenga este texto"
    )
    a = p.parse_args()
    if a.cmd == "export":
        export(a)
        return
    if a.cmd == "run":
        print(json.dumps(run_config(json.loads(a.config))))
        return
    cfgs = [c for c in matrix() if not a.only or a.only in c["name"]]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"hardware": hardware(), "results": []}
    for i, cfg in enumerate(cfgs, 1):
        res = run_isolated(cfg, script=__file__)
        report["results"].append(res)
        out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if "error" in res:
            print(
                f"[{i}/{len(cfgs)}] {cfg['name']} {cfg['device']} N={cfg['streams']} {cfg['source']}: ERROR {res['error']}"
            )
            continue
        print(
            f"[{i}/{len(cfgs)}] {res['name']:<26} {res['device']:<12} N={res['streams']} {res['source']:<5} "
            f"fps/stream={res['fps_per_stream']:5.1f} p50={res['cycle_ms_p50']:6.1f}ms "
            f"RAM={res['peak_private_gb']:.2f}GB GPU={res['peak_gpu_process_gb']:.2f}GB cpu={res['system_cpu_percent']:.0f}%",
            flush=True,
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    main()
