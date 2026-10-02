"""Descripciones `gst-launch` de los grafos de captura (solo texto, sin GStreamer).

- `cpu_graph`: una fuente decodificada en CPU hacia un appsink acotado; es el
  grafo de referencia para desarrollo en laptop sin GPU NVIDIA.
- `deepstream_graph`: grafo objetivo para la Jetson Orin Nano
  (NVDEC -> nvstreammux -> nvinfer FP16 -> nvtracker -> appsink).

IMPORTANTE: `deepstream_graph` NO ha sido validado en hardware; no hay Jetson
disponible. Es la especificación a verificar cuando llegue el dispositivo.
"""

from __future__ import annotations

from collections.abc import Sequence

from .shared_pipeline import SourceKind, StreamSource

DEFAULT_TRACKER_LIB = (
    "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
)
_APPSINK = "appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false"


def _usb_device(source: StreamSource) -> str:
    uri = source.uri.strip()
    for prefix in ("usb://", "usb:", "v4l2://"):
        if uri.startswith(prefix):
            uri = uri[len(prefix) :]
            break
    return f"/dev/video{uri}" if uri.isdigit() else uri


def _depay(codec: str) -> str:
    if codec not in ("h264", "h265"):
        raise ValueError(f"Códec no soportado: {codec!r} (use h264 o h265)")
    return f"rtp{codec}depay ! {codec}parse"


def cpu_graph(
    source: StreamSource, *, codec: str = "h264", latency_ms: int = 200
) -> str:
    """Grafo CPU: decodifica y entrega BGR a un appsink que descarta frames viejos."""
    if source.kind is SourceKind.RTSP:
        head = (
            f"rtspsrc location={source.uri} latency={latency_ms} drop-on-latency=true"
            f" ! {_depay(codec)} ! avdec_{codec}"
        )
    else:
        head = f"v4l2src device={_usb_device(source)}"
    return f"{head} ! videoconvert ! video/x-raw,format=BGR ! {_APPSINK}"


def deepstream_graph(
    sources: Sequence[StreamSource],
    *,
    engine_config: str,
    width: int = 1920,
    height: int = 1080,
    codec: str = "h264",
    tracker_lib: str = DEFAULT_TRACKER_LIB,
    latency_ms: int = 200,
) -> str:
    """Grafo DeepStream compartido: un decoder por fuente y un único nvinfer."""
    batch = len(sources)
    if not 1 <= batch <= 8:
        raise ValueError("El batch debe estar entre 1 y 8 fuentes (Orin Nano)")
    branches: list[str] = []
    for index, source in enumerate(sources):
        if source.kind is SourceKind.RTSP:
            branches.append(
                f"rtspsrc location={source.uri} latency={latency_ms} drop-on-latency=true"
                f" ! {_depay(codec)} ! nvv4l2decoder ! mux.sink_{index}"
            )
        else:
            branches.append(
                f"v4l2src device={_usb_device(source)} ! videoconvert ! nvvideoconvert"
                f" ! video/x-raw(memory:NVMM),format=NV12 ! mux.sink_{index}"
            )
    trunk = (
        f"nvstreammux name=mux batch-size={batch} width={width} height={height}"
        " live-source=1 batched-push-timeout=40000"
        f" ! nvinfer config-file-path={engine_config}"
        f" ! nvtracker ll-lib-file={tracker_lib}"
        f" ! {_APPSINK}"
    )
    return " ".join([*branches, trunk])
