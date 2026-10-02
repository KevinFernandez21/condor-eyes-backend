"""Tests del puente de tag BLE: protocolo, presencia, receptor y site map (sin BLE real)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tagbridge import (
    Advertisement,
    FakeAdvertisementSource,
    PresenceTracker,
    SiteMapError,
    TagReceiver,
    decode_tag_payload,
    encode_tag_payload,
    load_site_map,
    run_receiver,
)
from tagbridge.protocol import COMPANY_ID

TAG = "A1B2C3D4E5F6"
OTHER = "112233445566"


def adv(
    rssi: int, seq: int, t: float, tag: str = TAG, bat: int | None = 87
) -> Advertisement:
    return Advertisement(
        address="AA:BB:CC:DD:EE:FF",
        rssi=rssi,
        manufacturer_data={COMPANY_ID: encode_tag_payload(tag, seq, bat)},
        received_at=t,
    )


# --- protocolo -----------------------------------------------------------------


def test_payload_roundtrip():
    raw = encode_tag_payload(TAG, seq=1234, battery_pct=87)
    assert len(raw) == 14
    decoded = decode_tag_payload(raw)
    assert decoded is not None
    assert (decoded.tag, decoded.seq, decoded.battery_pct) == (TAG, 1234, 87)


def test_payload_battery_unknown_is_none():
    decoded = decode_tag_payload(encode_tag_payload(TAG, 1, None))
    assert decoded is not None and decoded.battery_pct is None


def test_payload_seq_is_uint32():
    decoded = decode_tag_payload(encode_tag_payload(TAG, 0xFFFFFFFF, 1))
    assert decoded is not None and decoded.seq == 0xFFFFFFFF
    with pytest.raises(ValueError):
        encode_tag_payload(TAG, 2**32, 1)


@pytest.mark.parametrize(
    "raw", [b"", b"CE", b"XX\x01" + bytes(11), b"CE\x09" + bytes(11)]
)
def test_payload_foreign_or_malformed_is_none(raw):
    assert decode_tag_payload(raw) is None


def test_payload_rejects_bad_tag():
    with pytest.raises(ValueError):
        encode_tag_payload("nothex", 1, 1)


# --- presencia -----------------------------------------------------------------


def make_tracker(**kw) -> PresenceTracker:
    base = {"enter_dbm": -70.0, "hysteresis_db": 6.0, "window": 1, "lost_after_s": 10.0}
    base.update(kw)
    return PresenceTracker(**base)


def test_presence_starts_unknown_then_decides_first_sample():
    t = make_tracker()
    assert t.state(0.0).status == "unknown"
    assert t.update(-60, 0.0).status == "inside"
    t2 = make_tracker()
    assert t2.update(-85, 0.0).status == "outside"


def test_presence_hysteresis_prevents_flapping():
    t = make_tracker()
    assert t.update(-60, 0.0).status == "inside"
    # banda de histéresis: -70 > rssi >= -76 se mantiene dentro
    assert t.update(-74, 1.0).status == "inside"
    assert t.update(-77, 2.0).status == "outside"
    # una vez fuera, hay que superar el umbral de entrada, no el de salida
    assert t.update(-72, 3.0).status == "outside"
    assert t.update(-69, 4.0).status == "inside"


def test_presence_smoothing_uses_mean_of_window():
    t = make_tracker(window=3)
    t.update(-60, 0.0)
    t.update(-60, 1.0)
    st = t.update(-90, 2.0)
    assert st.smoothed_dbm == pytest.approx(-70.0)


def test_presence_confidence_grows_with_margin_and_is_clamped():
    t = make_tracker(margin_full_db=10.0)
    near = t.update(-69, 0.0)
    far = make_tracker(margin_full_db=10.0).update(-50, 0.0)
    assert 0.0 <= near.confidence < far.confidence <= 1.0
    assert far.confidence == 1.0


def test_presence_lost_after_timeout_is_outside_without_confidence():
    t = make_tracker(lost_after_s=5.0)
    t.update(-60, 0.0)
    assert t.state(4.0).status == "inside"
    lost = t.state(6.0)
    assert (lost.status, lost.reason, lost.confidence) == ("outside", "lost", 0.0)
    assert t.update(-60, 7.0).status == "inside"


def test_presence_rejects_inconsistent_params():
    with pytest.raises(ValueError):
        PresenceTracker(enter_dbm=-70, hysteresis_db=-1)
    with pytest.raises(ValueError):
        PresenceTracker(enter_dbm=-70, hysteresis_db=3, window=0)


# --- receptor ------------------------------------------------------------------


def make_receiver(**kw) -> TagReceiver:
    return TagReceiver(enrolled={TAG}, node_id="N0001", tracker=make_tracker(), **kw)


def test_receiver_emits_payload_v1_for_enrolled_tag():
    rx = make_receiver()
    ev = rx.on_advertisement(adv(-63, 10, t=1767268800.25, bat=90))
    assert ev is not None
    assert ev.observation == {
        "v": 1,
        "ch": "ble",
        "tag": TAG,
        "node": "N0001",
        "rssi": -63,
        "ts": 1767268800250,
        "seq": 10,
        "bat": 90,
    }
    assert ev.state.status == "inside"
    json.dumps(ev.observation)


def test_receiver_omits_bat_when_unknown():
    ev = make_receiver().on_advertisement(adv(-63, 1, 1.0, bat=None))
    assert ev is not None and ev.observation is not None and "bat" not in ev.observation


def test_receiver_ignores_non_enrolled_and_foreign_devices_silently(caplog, capsys):
    rx = make_receiver()
    foreign = Advertisement(
        "00:11:22:33:44:55", -50, {0x004C: b"\x02\x15" + bytes(21)}, 1.0
    )
    nameless = Advertisement("00:11:22:33:44:56", -50, {}, 1.0)
    with caplog.at_level("DEBUG"):
        assert rx.on_advertisement(adv(-50, 1, 1.0, tag=OTHER)) is None
        assert rx.on_advertisement(foreign) is None
        assert rx.on_advertisement(nameless) is None
    out = capsys.readouterr()
    assert caplog.text == "" and out.out == "" and out.err == ""
    assert rx.stats["ignored"] == 3


def test_receiver_drops_duplicate_and_older_seq():
    rx = make_receiver()
    assert rx.on_advertisement(adv(-60, 5, 1.0)) is not None
    assert rx.on_advertisement(adv(-61, 5, 1.1)) is None
    assert rx.on_advertisement(adv(-62, 4, 1.2)) is None
    assert rx.on_advertisement(adv(-62, 6, 1.3)) is not None
    assert rx.stats["duplicates"] == 2


def test_receiver_accepts_seq_wraparound():
    rx = make_receiver()
    assert rx.on_advertisement(adv(-60, 0xFFFFFFFF, 1.0)) is not None
    assert rx.on_advertisement(adv(-60, 0, 1.1)) is not None


def test_receiver_enrollment_is_case_insensitive():
    rx = TagReceiver(enrolled={TAG.lower()}, node_id="N0001", tracker=make_tracker())
    assert rx.on_advertisement(adv(-60, 1, 1.0)) is not None


def test_receiver_rejects_invalid_node_id():
    with pytest.raises(ValueError):
        TagReceiver(enrolled={TAG}, node_id="bad node!", tracker=make_tracker())


def test_receiver_rejects_invalid_enrolled_tag():
    with pytest.raises(ValueError):
        TagReceiver(enrolled={"xyz"}, node_id="N0001", tracker=make_tracker())


def test_run_receiver_with_fake_source_reports_states_and_lost():
    clock = {"t": 0.0}
    source = FakeAdvertisementSource(
        [adv(-60, 1, 0.0), adv(-50, 1, 0.1, tag=OTHER), adv(-61, 2, 1.0), None, None],
        clock=clock,
    )
    tracker = make_tracker(lost_after_s=1.5)
    rx = TagReceiver(
        enrolled={TAG}, node_id="N0001", tracker=tracker, clock=lambda: clock["t"]
    )
    seen = []
    asyncio.run(run_receiver(source, rx, seen.append, tick_s=1.0))
    observed = [e for e in seen if e.observation is not None]
    assert [e.state.status for e in observed] == ["inside", "inside"]
    assert all(e.observation["tag"] == TAG for e in observed)  # type: ignore[index]
    assert seen[-1].state.reason == "lost" and seen[-1].observation is None


# --- site map ------------------------------------------------------------------

SITE = """
[[cameras]]
id = "laptop-webcam"
source = 0

[[receivers]]
id = "N0001"
kind = "pc-ble"
zone = "lab"

[[zones]]
id = "lab"
name = "Laboratorio"
receiver = "N0001"
camera = "laptop-webcam"
region = [0.1, 0.2, 0.9, 0.8]
enter_dbm = -70
hysteresis_db = 6
calibrated = false
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "site_map.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_site_map_loads_and_links_zone_to_camera_region(tmp_path):
    sm = load_site_map(write(tmp_path, SITE))
    z = sm.zone("lab")
    assert z.receiver == "N0001" and z.camera == "laptop-webcam"
    assert z.region == (0.1, 0.2, 0.9, 0.8)
    assert z.calibrated is False
    assert sm.receiver("N0001").zone == "lab"


def test_site_map_unknown_camera_is_error(tmp_path):
    with pytest.raises(SiteMapError, match="cámara"):
        load_site_map(
            write(tmp_path, SITE.replace('camera = "laptop-webcam"', 'camera = "nope"'))
        )


def test_site_map_unknown_receiver_is_error(tmp_path):
    with pytest.raises(SiteMapError, match="receptor"):
        load_site_map(
            write(tmp_path, SITE.replace('receiver = "N0001"', 'receiver = "N9"'))
        )


def test_site_map_bad_region_is_error(tmp_path):
    with pytest.raises(SiteMapError, match="región"):
        load_site_map(
            write(
                tmp_path, SITE.replace("[0.1, 0.2, 0.9, 0.8]", "[0.9, 0.2, 0.1, 0.8]")
            )
        )


def test_site_map_duplicate_zone_is_error(tmp_path):
    zone_block = "[[zones]]" + SITE.split("[[zones]]")[1]
    with pytest.raises(SiteMapError, match="duplicad"):
        load_site_map(write(tmp_path, SITE + zone_block))


def test_repo_site_map_is_valid():
    sm = load_site_map(
        Path(__file__).resolve().parents[1] / "configs" / "site_map.toml"
    )
    assert sm.zones and all(sm.camera(z.camera) for z in sm.zones.values())


def test_cli_format_event_shows_rssi_and_status():
    from tagbridge.cli import format_event

    ev = make_receiver().on_advertisement(adv(-63, 10, t=1767268800.25))
    assert ev is not None
    line = format_event(ev)
    assert "-63" in line and "inside" in line and "seq=10" in line
    assert "sin señal" in format_event(make_receiver().tick())
