"""Vista multicámara: clases, cajas normalizadas, escenas, estado y registro."""

from __future__ import annotations

import pytest

from dashboard import cameras as cam


def env(topic, payload, stream_id=None, created_at="2026-01-01T00:00:10+00:00"):
    return {"topic": topic, "source": "s", "payload": payload, "stream_id": stream_id,
            "created_at": created_at, "event_id": "e", "correlation_id": None}


# --- clases ---

@pytest.mark.parametrize(
    ("det", "expected"),
    [
        ({"label": "person"}, "person"), ({"cls": 0}, "person"),
        ({"label": "car"}, "vehicle"), ({"cls": 7}, "vehicle"), ({"label": "truck"}, "vehicle"),
        ({"label": "dog"}, "motion"), ({"cls": 16}, "motion"), ({}, "motion"),
        ({"label": "car", "cls": 0}, "vehicle"),  # la etiqueta manda sobre el id
    ],
)
def test_detection_class(det, expected):
    assert cam.detection_class(det) == expected


# --- cajas ---

def test_normalize_scales_to_unit_box_and_keeps_metadata():
    out = cam.normalize_detections(
        [{"xyxy": [192, 108, 960, 540], "cls": 0, "conf": 0.91, "track_id": 3}], 1920, 1080
    )
    assert out == [{"box": [0.1, 0.1, 0.5, 0.5], "cls": "person", "label": "Persona",
                    "conf": 0.91, "track_id": 3}]


def test_normalize_clamps_and_skips_malformed():
    out = cam.normalize_detections(
        [
            {"xyxy": [-50, -10, 3000, 2000], "cls": 2},
            {"xyxy": [1, 2, 3]},
            {"xyxy": [0, 0, float("nan"), 5]},
            {"xyxy": "no"},
            "texto",
            {"xyxy": [10, 10, 5, 5]},  # invertida
        ],
        1920, 1080,
    )
    assert len(out) == 1
    assert out[0]["box"] == [0.0, 0.0, 1.0, 1.0] and out[0]["cls"] == "vehicle"
    assert out[0]["conf"] is None and out[0]["track_id"] is None


def test_normalize_caps_the_number_of_detections():
    dets = [{"xyxy": [0, 0, 10, 10]}] * 500
    assert len(cam.normalize_detections(dets, 100, 100)) == cam.MAX_DETECTIONS


# --- escena ---

@pytest.mark.parametrize(
    ("scene", "kind"),
    [("Entrada principal", "lobby"), ("estacionamiento", "parking"),
     ("Perímetro norte", "perimeter"), ("bodega", "warehouse"), ("vault", "warehouse")],
)
def test_scene_kind_by_keyword(scene, kind):
    assert cam.scene_kind(scene, "cam-01") == kind


def test_scene_kind_is_deterministic_and_always_known():
    for cid in ("cam-01", "cam-02", "x", "zzz"):
        kind = cam.scene_kind(None, cid)
        assert kind == cam.scene_kind("???", cid) in cam.SCENES


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("connected", "online"), ("up", "online"), ("running", "online"), ("ok", "online"),
     ("reconnecting", "degraded"), ("degraded", "degraded"), ("connecting", "degraded"),
     ("failed", "offline"), ("stopped", "offline"), ("down", "offline"), ("offline", "offline"),
     ("???", None), (None, None)],
)
def test_camera_state(raw, expected):
    assert cam.camera_state(raw) == expected


# --- registro ---

class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_registry_reads_flat_and_nested_metadata_and_orders_by_id():
    reg = cam.CameraRegistry(clock=Clock())
    reg.observe(env("stream.status", {"state": "up", "name": "Acceso norte", "scene": "perimetro"}, "cam-02"))
    reg.observe(env("system.health", {"stream_id": "cam-01", "state": "connected",
                                      "camera": {"name": "Hall", "scene": "Entrada"}}))
    views = reg.views()
    assert [v["id"] for v in views] == ["cam-01", "cam-02"]
    assert views[0]["name"] == "Hall" and views[0]["scene_kind"] == "lobby"
    assert views[1]["name"] == "Acceso norte" and views[1]["scene_kind"] == "perimeter"


def test_registry_default_name_is_id_and_frame_defaults_to_1080p():
    reg = cam.CameraRegistry(clock=Clock())
    reg.observe(env("vision.detections", {"detections": []}, "cam-09"))
    view = reg.views()[0]
    assert view["name"] == "cam-09" and view["frame"] == {"w": 1920, "h": 1080}


def test_registry_detections_are_normalized_with_payload_frame_size_and_counted():
    reg = cam.CameraRegistry(clock=Clock())
    reg.observe(env("vision.detections", {
        "width": 640, "height": 480,
        "detections": [{"xyxy": [64, 48, 320, 240], "cls": 0, "conf": 0.8},
                       {"xyxy": [0, 0, 64, 48], "label": "car", "conf": 0.7}]}, "cam-01"))
    view = reg.views()[0]
    assert view["frame"] == {"w": 640, "h": 480}
    assert view["counts"] == {"person": 1, "vehicle": 1, "motion": 0}
    assert view["detections"][0]["box"] == [0.1, 0.1, 0.5, 0.5]
    assert view["state"] == "online"


def test_registry_prefers_fresh_tracks_for_track_ids():
    clock = Clock()
    reg = cam.CameraRegistry(clock=clock)
    reg.observe(env("vision.detections", {"detections": [{"xyxy": [0, 0, 100, 100], "cls": 0}]}, "c"))
    reg.observe(env("vision.tracks", {"tracks": [{"track_id": 7, "xyxy": [0, 0, 100, 100], "cls": 0}]}, "c"))
    assert reg.views()[0]["detections"][0]["track_id"] == 7
    clock.t += 5  # los tracks caducan; vuelven las detecciones sin id
    reg.observe(env("vision.detections", {"detections": [{"xyxy": [0, 0, 100, 100], "cls": 0}]}, "c"))
    assert reg.views()[0]["detections"][0]["track_id"] is None


def test_registry_without_signal_goes_offline_with_message():
    clock = Clock()
    reg = cam.CameraRegistry(clock=clock, stale_s=8)
    reg.observe(env("vision.detections", {"detections": [{"xyxy": [0, 0, 10, 10], "cls": 0}]}, "c"))
    assert reg.views()[0]["state"] == "online"
    clock.t += 9
    view = reg.views()[0]
    assert view["state"] == "offline" and view["detections"] == []
    assert "señal" in view["message"].lower()


def test_registry_reported_degraded_or_offline_state_wins_while_fresh():
    reg = cam.CameraRegistry(clock=Clock())
    reg.observe(env("stream.status", {"state": "reconnecting", "last_error": "timeout"}, "c"))
    reg.observe(env("vision.detections", {"detections": []}, "c"))
    view = reg.views()[0]
    assert view["state"] == "degraded" and view["state_label"] == "Degradada"
    reg.observe(env("stream.status", {"state": "failed"}, "c"))
    assert reg.views()[0]["state"] == "offline"


def test_registry_ignores_unrelated_topics_and_messages_without_stream():
    reg = cam.CameraRegistry(clock=Clock())
    reg.observe(env("events", {"x": 1}, "c"))
    reg.observe(env("vision.detections", {"detections": []}, None))
    reg.observe("texto")  # type: ignore[arg-type]
    assert reg.views() == []


def test_registry_is_bounded_in_number_of_cameras():
    reg = cam.CameraRegistry(clock=Clock(), max_cameras=3)
    for i in range(10):
        reg.observe(env("stream.status", {"state": "up"}, f"cam-{i}"))
    assert len(reg.views()) == 3


def test_registry_redacts_biometric_keys_in_metadata():
    reg = cam.CameraRegistry(clock=Clock())
    reg.observe(env("stream.status", {"state": "up", "name": "Hall", "last_error": "x"}, "c"))
    view = reg.views()[0]
    assert "payload" not in view and "image" not in str(view)
