"""Cableado mínimo del servidor del dashboard (sin red hacia la API)."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from dashboard import DashboardConfig, DashboardConfigurationError, create_app
from dashboard.server import check_bind, is_local_host
from dashboard.state import DashboardState


def _client(state=None):
    app = create_app(DashboardConfig(site_map_path=None), state=state, start_clients=False)
    return TestClient(app)


def test_index_serves_the_page_with_viewport_and_no_external_assets():
    html = _client().get("/").text
    assert 'name="viewport"' in html
    assert not re.search(r'(src|href)="https?://', html)  # sin CDN: funciona sin internet


def test_static_assets_are_served():
    client = _client()
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


@pytest.mark.parametrize("name", ["common.js", "agents.js", "cameras.js", "app.js"])
def test_scripts_are_served_and_never_insert_html(name):
    response = _client().get(f"/static/{name}")
    assert response.status_code == 200
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in response.text


def test_index_has_both_views_and_navigation():
    html = _client().get("/").text
    for needle in ('id="view-agents"', 'id="view-cams"', 'data-view="agents"', 'data-view="cams"',
                   'id="cam-grid"', 'id="tl"', 'id="review-log"', 'id="toasts"'):
        assert needle in html


def test_api_state_includes_multicamera_fields():
    body = _client().get("/api/state").json()
    assert {"cameras", "timeline", "components"} <= set(body)
    assert body["timeline"]["series"] == {"all": []}


def test_api_state_shape_and_topic_filter():
    state = DashboardState()
    for i, topic in enumerate(("events", "vision.tracks")):
        state.on_ws_message(
            {"type": "envelope", "data": {"topic": topic, "source": "s", "payload": {},
                                           "event_id": str(i), "created_at": "2026-01-01T00:00:00+00:00"}}
        )
    body = _client(state).get("/api/state", params={"topics": "events"}).json()
    assert [e["topic"] for e in body["feed"]] == ["events"]
    assert {"graph", "alerts", "health", "site", "camera", "api", "ws", "lag"} <= set(body)
    assert body["camera"]["available"] is False
    assert "pipeline" in body["camera"]["message"]


def test_api_state_without_api_reports_offline_without_crashing():
    body = _client().get("/api/state").json()
    assert body["health"]["status"] == "offline"
    assert body["graph"]["nodes"]


def test_api_state_rejects_out_of_range_limit():
    assert _client().get("/api/state", params={"limit": 0}).status_code == 422


def test_dashboard_has_no_video_route():
    paths = {getattr(r, "path", "") for r in create_app(start_clients=False).routes}
    assert not any("video" in p or "mjpeg" in p or "frame" in p for p in paths)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_bind_is_allowed(host):
    check_bind(host, allow_lan=False)
    assert is_local_host(host)


def test_lan_bind_requires_explicit_flag():
    with pytest.raises(DashboardConfigurationError, match="autenticación"):
        check_bind("0.0.0.0", allow_lan=False)
    check_bind("0.0.0.0", allow_lan=True)


def test_index_is_never_cached_and_every_asset_is_versioned_by_content_hash():
    response = _client().get("/")
    assert response.headers["cache-control"] == "no-store"
    refs = re.findall(r'(?:src|href)="(/static/[^"]+)"', response.text)
    assert len(refs) >= 5
    for ref in refs:
        match = re.fullmatch(r"(/static/[\w.\-]+)\?v=([0-9a-f]{10})", ref)
        assert match, f"recurso sin versión: {ref}"
        from dashboard.server import STATIC_DIR, asset_version

        assert match.group(2) == asset_version(STATIC_DIR / match.group(1).removeprefix("/static/"))


def test_every_static_file_referenced_by_the_template_exists():
    from dashboard.server import STATIC_DIR

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for name in re.findall(r'(?:src|href)="/static/([^"]+)"', html):
        assert (STATIC_DIR / name).is_file()


def test_version_changes_when_content_changes(tmp_path):
    from dashboard.server import asset_version

    f = tmp_path / "a.js"
    f.write_text("1", encoding="utf-8")
    first = asset_version(f)
    f.write_text("2", encoding="utf-8")
    assert asset_version(f) != first


def test_static_files_must_revalidate_and_versioned_url_resolves():
    client = _client()
    response = client.get("/static/app.js?v=abc")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert "etag" in response.headers
