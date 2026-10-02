"""Tests del puente de tag BLE: protocolo, presencia, receptor y site map (sin BLE real)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from location import (
    Evidence,
    InMemoryPersonnelRepository,
    LocationService,
    PersonnelRecord,
    Pseudonymizer,
    RejectReason,
    ZoneEstimate,
    load_config,
)
from location.models import EstimateStatus, UnknownReason
from tagbridge import (
    Advertisement,
    FakeAdvertisementSource,
    PresenceView,
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
T0 = 1767268800.0
LOCATION_CFG = Path(__file__).resolve().parents[1] / "configs" / "location.toml"


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


# --- vista de presencia (sobre la salida de LocationService) -------------------


def estimate(rssi: float | None, status=EstimateStatus.LOCATED, reason=None, conf=0.9):
    now = datetime.fromtimestamp(T0, UTC)
    ev = ()
    if rssi is not None:
        ev = (Evidence("N0001", "lobby", rssi, 3, now, now),)
    return ZoneEstimate(
        tag_ref="tag-x",
        status=status,
        zone_id="lobby" if status is EstimateStatus.LOCATED else None,
        confidence=conf,
        computed_at=now,
        last_evidence_at=now if ev else None,
        evidence=ev,
        unknown_reason=reason,
    )


def make_view(**kw) -> PresenceView:
    return PresenceView(**({"enter_dbm": -70.0, "hysteresis_db": 6.0} | kw))


def test_view_never_seen_is_unknown():
    st = make_view().update(
        estimate(None, EstimateStatus.UNKNOWN, UnknownReason.TAG_MISSING)
    )
    assert st.status == "unknown" and st.reason == "tag_missing"


def test_view_decides_on_first_estimate():
    assert make_view().update(estimate(-60)).status == "inside"
    assert make_view().update(estimate(-85)).status == "outside"


def test_view_hysteresis_prevents_flapping():
    v = make_view()
    assert v.update(estimate(-60)).status == "inside"
    assert v.update(estimate(-74)).status == "inside"  # banda de histéresis
    assert v.update(estimate(-77)).status == "outside"
    assert (
        v.update(estimate(-72)).status == "outside"
    )  # hay que superar el umbral de entrada
    assert v.update(estimate(-69)).status == "inside"


def test_view_uses_strongest_node_and_service_confidence():
    now = datetime.fromtimestamp(T0, UTC)
    base = estimate(-60, conf=0.42)
    extra = Evidence("N0002", "lobby", -90.0, 3, now, now)
    est = replace(base, evidence=(extra, *base.evidence))
    st = make_view().update(est)
    assert st.smoothed_dbm == -60 and st.confidence == 0.42 and st.zone_id == "lobby"


def test_view_stale_evidence_is_outside_and_resets_latch():
    v = make_view()
    v.update(estimate(-60))
    lost = v.update(
        estimate(None, EstimateStatus.UNKNOWN, UnknownReason.STALE_EVIDENCE)
    )
    assert (lost.status, lost.reason) == ("outside", "stale_evidence")
    # al volver decide como la primera vez: -72 está por debajo del umbral de entrada
    assert v.update(estimate(-72)).status == "outside"


def test_view_rejects_inconsistent_params():
    with pytest.raises(ValueError):
        PresenceView(enter_dbm=-70, hysteresis_db=-1)


# --- receptor + LocationService ------------------------------------------------


def make_service(clock):
    pseudo = Pseudonymizer(b"test-key")
    repo = InMemoryPersonnelRepository(pseudo)
    repo.enroll(TAG, PersonnelRecord("p1", "Persona", frozenset({"lobby"})))
    cfg = load_config(LOCATION_CFG)
    return LocationService(
        cfg, repo, pseudo, clock=lambda: datetime.fromtimestamp(clock["t"], UTC)
    )


def make_receiver(clock=None):
    clock = clock if clock is not None else {"t": T0}
    svc = make_service(clock)
    rx = TagReceiver(
        service=svc,
        enrolled={TAG},
        node_id="N0001",
        view=make_view(),
        clock=lambda: clock["t"],
    )
    return rx, svc, clock


def test_receiver_emits_payload_v1_and_feeds_location_service():
    rx, svc, _ = make_receiver()
    ev = rx.on_advertisement(adv(-63, 10, t=T0 + 0.25, bat=90))
    assert ev is not None
    assert ev.observation == {
        "v": 1,
        "ch": "ble",
        "tag": TAG,
        "node": "N0001",
        "rssi": -63,
        "ts": round((T0 + 0.25) * 1000),
        "seq": 10,
        "bat": 90,
    }
    json.dumps(ev.observation)
    assert ev.ingest is not None and ev.ingest.accepted
    assert svc.accepted_count == 1
    assert ev.state.status == "inside" and ev.state.zone_id == "lobby"
    assert 0.0 < ev.state.confidence <= 1.0


def test_receiver_omits_bat_when_unknown():
    rx, _, _ = make_receiver()
    ev = rx.on_advertisement(adv(-63, 1, T0, bat=None))
    assert ev is not None and ev.observation is not None
    assert "bat" not in ev.observation


def test_receiver_ignores_non_enrolled_and_foreign_devices_silently(caplog, capsys):
    rx, svc, _ = make_receiver()
    foreign = Advertisement(
        "00:11:22:33:44:55", -50, {0x004C: b"\x02\x15" + bytes(21)}, T0
    )
    nameless = Advertisement("00:11:22:33:44:56", -50, {}, T0)
    with caplog.at_level("DEBUG"):
        assert rx.on_advertisement(adv(-50, 1, T0, tag=OTHER)) is None
        assert rx.on_advertisement(foreign) is None
        assert rx.on_advertisement(nameless) is None
    out = capsys.readouterr()
    assert caplog.text == "" and out.out == "" and out.err == ""
    assert rx.ignored == 3
    assert not svc.stats and svc.accepted_count == 0


def test_receiver_duplicate_and_old_seq_are_rejected_by_location_service():
    rx, svc, _ = make_receiver()
    assert rx.on_advertisement(adv(-60, 5, T0)) is not None
    dup = rx.on_advertisement(adv(-61, 5, T0 + 0.1))
    old = rx.on_advertisement(adv(-62, 4, T0 + 0.2))
    assert dup is not None and dup.ingest is not None
    assert dup.ingest.reason is RejectReason.DUPLICATE
    assert old is not None and old.ingest is not None
    assert old.ingest.reason is RejectReason.REPLAY
    assert (
        svc.stats[RejectReason.DUPLICATE] == 1 and svc.stats[RejectReason.REPLAY] == 1
    )


def test_receiver_accepts_seq_wraparound():
    rx, svc, _ = make_receiver()
    rx.on_advertisement(adv(-60, 0xFFFFFFFF, T0))
    rx.on_advertisement(adv(-60, 0, T0 + 0.1))
    assert svc.accepted_count == 2


def test_receiver_tick_after_silence_is_outside_stale_evidence():
    rx, _, clock = make_receiver()
    rx.on_advertisement(adv(-60, 1, T0))
    clock["t"] = T0 + 20  # > evidence_max_age_s, < node_timeout_s
    ev = rx.tick()
    assert ev.observation is None
    assert (ev.state.status, ev.state.reason) == ("outside", "stale_evidence")
    clock["t"] = T0 + 60  # el único nodo tampoco da señales de vida
    assert rx.tick().state.reason == "node_outage"


def test_receiver_tick_before_any_advert_is_unknown():
    rx, _, _ = make_receiver()
    assert rx.tick().state.status == "unknown"


def test_receiver_enrollment_is_case_insensitive():
    _, svc, clock = make_receiver()
    rx2 = TagReceiver(
        service=svc,
        enrolled={TAG.lower()},
        node_id="N0001",
        view=make_view(),
        clock=lambda: clock["t"],
    )
    assert rx2.on_advertisement(adv(-60, 1, T0)) is not None


def test_receiver_rejects_invalid_node_id():
    _, svc, _ = make_receiver()
    with pytest.raises(ValueError):
        TagReceiver(service=svc, enrolled={TAG}, node_id="bad node!", view=make_view())


def test_receiver_rejects_invalid_enrolled_tag():
    _, svc, _ = make_receiver()
    with pytest.raises(ValueError):
        TagReceiver(service=svc, enrolled={"xyz"}, node_id="N0001", view=make_view())


def test_run_receiver_with_fake_source_reports_states_and_lost():
    clock = {"t": T0}
    source = FakeAdvertisementSource(
        [
            adv(-60, 1, T0),
            adv(-50, 1, T0 + 0.1, tag=OTHER),
            adv(-61, 2, T0 + 1.0),
            *[None] * 20,
        ],
        clock=clock,
    )
    rx, _, _ = make_receiver(clock)
    seen = []
    asyncio.run(run_receiver(source, rx, seen.append, tick_s=1.0))
    observed = [e for e in seen if e.observation is not None]
    assert [e.state.status for e in observed] == ["inside", "inside"]
    assert all(e.observation["tag"] == TAG for e in observed)  # type: ignore[index]
    assert seen[-1].state.reason == "stale_evidence" and seen[-1].observation is None


def test_cli_format_event_shows_rssi_status_and_zone():
    from tagbridge.cli import format_event

    rx, _, _ = make_receiver()
    ev = rx.on_advertisement(adv(-63, 10, t=T0))
    assert ev is not None
    line = format_event(ev)
    assert "-63" in line and "inside" in line and "seq=10" in line and "lobby" in line
    assert "sin señal" in format_event(rx.tick())


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


def test_repo_site_map_matches_location_config():
    root = Path(__file__).resolve().parents[1] / "configs"
    sm = load_site_map(root / "site_map.toml")
    loc = load_config(root / "location.toml")
    for zone in sm.zones.values():
        assert loc.zone_of_node(zone.receiver) == zone.id
