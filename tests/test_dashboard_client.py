"""Estado del dashboard, backoff del WS y configuración."""

from __future__ import annotations

import pytest

from dashboard.client import Backoff
from dashboard.config import DashboardConfig
from dashboard.state import DashboardState


def env(topic, payload=None, eid="e", **extra):
    base = {"topic": topic, "source": "s", "payload": payload or {}, "stream_id": None,
            "created_at": "2026-01-01T00:00:00+00:00", "event_id": eid,
            "correlation_id": None, "causation_id": None}
    base.update(extra)
    return base


def test_backoff_grows_and_caps_and_resets():
    b = Backoff(initial=1.0, factor=2.0, maximum=8.0)
    assert [b.next() for _ in range(5)] == [1.0, 2.0, 4.0, 8.0, 8.0]
    b.reset()
    assert b.next() == 1.0


def test_config_from_env_defaults_and_overrides():
    cfg = DashboardConfig.from_env({})
    assert cfg.api_url == "http://127.0.0.1:8000"
    assert cfg.token is None
    cfg = DashboardConfig.from_env(
        {"CONDOR_API_URL": "http://10.0.0.5:9000/", "CONDOR_API_TOKEN": "t",
         "CONDOR_SITE_MAP": "x.toml"}
    )
    assert cfg.api_url == "http://10.0.0.5:9000"
    assert cfg.token == "t" and cfg.site_map_path == "x.toml"


def test_config_ws_url_has_scheme_and_never_carries_the_token():
    cfg = DashboardConfig(api_url="https://h:1", token="secreto")
    assert cfg.ws_url() == "wss://h:1/ws"
    assert DashboardConfig(api_url="http://h:1/").ws_url() == "ws://h:1/ws"
    assert "secreto" not in cfg.ws_url()


def test_config_headers_carry_the_bearer_token_for_http_and_ws():
    assert DashboardConfig(token="t").headers() == {"Authorization": "Bearer t"}
    assert DashboardConfig().headers() == {}


@pytest.mark.parametrize("raw", ["", "   "])
def test_config_blank_token_is_none(raw):
    assert DashboardConfig.from_env({"CONDOR_API_TOKEN": raw}).token is None


def test_state_feed_is_bounded_and_newest_first():
    state = DashboardState(feed_size=3)
    for i in range(5):
        state.on_ws_message({"type": "envelope", "data": env("events", {"i": i}, eid=str(i))})
    snap = state.snapshot()
    assert [e["event_id"] for e in snap["feed"]] == ["4", "3", "2"]


def test_state_lag_message_is_counted_not_fed():
    state = DashboardState()
    state.on_ws_message({"type": "hello", "topics": ["events"], "queue_size": 256})
    state.on_ws_message({"type": "lag", "dropped": 12, "dropped_total": 40})
    snap = state.snapshot()
    assert snap["feed"] == []
    assert snap["lag"]["dropped_total"] == 40
    assert snap["ws"]["connected"] is True


def test_state_ws_disconnect_is_reflected():
    state = DashboardState()
    state.on_ws_message({"type": "hello", "topics": []})
    state.on_ws_closed("cierre 1006")
    snap = state.snapshot()
    assert snap["ws"]["connected"] is False and snap["ws"]["error"] == "cierre 1006"


def test_state_ignores_malformed_messages():
    state = DashboardState()
    state.on_ws_message({"type": "envelope"})
    state.on_ws_message({"nope": 1})
    state.on_ws_message("texto")  # type: ignore[arg-type]
    assert state.snapshot()["feed"] == []


def test_state_snapshot_filters_feed_by_topic():
    state = DashboardState()
    state.on_ws_message({"type": "envelope", "data": env("events", eid="1")})
    state.on_ws_message({"type": "envelope", "data": env("vision.tracks", eid="2")})
    snap = state.snapshot(topics={"events"})
    assert [e["event_id"] for e in snap["feed"]] == ["1"]


def test_state_builds_alerts_from_decisions_and_tags_from_feed():
    state = DashboardState()
    state.on_ws_message({"type": "envelope", "data": env(
        "events", {"decision_id": "d1", "outcome": "alert", "requires_operator": True}, "a")})
    state.on_ws_message({"type": "envelope", "data": env(
        "events", {"tag_ref": "tg", "status": "located", "zone_id": "z", "confidence": 0.5}, "b")})
    snap = state.snapshot()
    assert [a["decision_id"] for a in snap["alerts"]] == ["d1"]
    assert snap["site"] is None  # sin mapa configurado
    assert snap["tags"][0]["tag_ref"] == "tg"


def test_state_poll_failure_marks_api_offline_but_keeps_last_data():
    state = DashboardState()
    state.on_poll({"agents": [], "topics": [], "health": {"status": "ok"}, "decisions": []})
    state.on_poll_error("conexión rechazada")
    snap = state.snapshot()
    assert snap["api"]["connected"] is False
    assert snap["api"]["error"] == "conexión rechazada"
    assert snap["health"]["status"] == "offline"
    assert snap["graph"]["nodes"]  # el grafo sigue ahí


def test_state_poll_seeds_alerts_without_duplicates():
    state = DashboardState()
    d = env("events", {"decision_id": "d1", "outcome": "alert"}, "x")
    state.on_ws_message({"type": "envelope", "data": d})
    state.on_poll({"agents": [], "topics": [], "health": {"status": "ok"}, "decisions": [d]})
    assert len(state.snapshot()["alerts"]) == 1
