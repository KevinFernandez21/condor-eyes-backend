"""Funciones puras del dashboard: grafo, feed, alertas, mapa de sitio, salud."""

from __future__ import annotations

import pytest

from dashboard import viewmodel as vm


def agent(name, role, state="running", **extra):
    base = {
        "name": name, "role": role, "instance": None, "state": state,
        "processed": 3, "duplicates": 0, "failures": 0, "retries": 0,
        "last_error": None, "last_heartbeat": None, "restarts": 0, "queue_depth": 0,
    }
    base.update(extra)
    return base


def topic(name, rate=0.0, drops=0, count=0):
    return {"topic": name, "count": count, "rate_per_s": rate, "window_s": 10,
            "drops": drops, "last_message_at": None}


ROUTES = [
    vm.RouteSpec("ingest", consumes=("system.commands",), publishes=("stream.status",)),
    vm.RouteSpec("inference", consumes=(), publishes=("vision.detections",)),
    vm.RouteSpec("tracker", consumes=("vision.detections",), publishes=("vision.tracks",)),
]


# --- estado de agentes ---

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("running", "ok"), ("WorkerState.RUNNING", "ok"), ("failed", "failed"),
        ("starting", "degraded"), ("stopping", "degraded"),
        ("stopped", "stopped"), ("created", "stopped"),
    ],
)
def test_agent_state_maps_worker_state(raw, expected):
    assert vm.agent_state(agent("a", "a", raw)) == expected


def test_running_agent_with_failures_is_degraded():
    assert vm.agent_state(agent("a", "a", failures=2)) == "degraded"


# --- grafo ---

def test_graph_edges_follow_topics_and_carry_stats():
    graph = vm.build_graph(
        [agent("ingest", "ingest"), agent("tracker", "tracker")],
        [topic("vision.detections", rate=4.5, drops=2)],
        ROUTES,
    )
    edge = next(e for e in graph["edges"] if e["topic"] == "vision.detections")
    assert (edge["source"], edge["target"]) == ("inference", "tracker")
    assert edge["rate_per_s"] == 4.5 and edge["drops"] == 2
    assert "4.5" in edge["label"] and "2" in edge["label"]


def test_graph_node_without_agent_data_is_unknown():
    graph = vm.build_graph([agent("ingest", "ingest")], [], ROUTES)
    states = {n["id"]: n["state"] for n in graph["nodes"]}
    assert states["ingest"] == "ok"
    assert states["tracker"] == "unknown"


def test_graph_role_aggregates_instances_worst_state_wins():
    agents = [
        agent("ingest/cam-1", "ingest", instance="cam-1"),
        agent("ingest/cam-2", "ingest", "failed", instance="cam-2", last_error="boom"),
    ]
    graph = vm.build_graph(agents, [], ROUTES)
    node = next(n for n in graph["nodes"] if n["id"] == "ingest")
    assert node["state"] == "failed"
    assert node["instances"] == 2
    assert "boom" in node["detail"]


def test_graph_includes_planned_roles_fusion_location_identity_actuation():
    graph = vm.build_graph([], [], vm.default_routes())
    ids ={n["id"] for n in graph["nodes"]}
    assert {"fusion", "location", "identity", "actuation"} <= ids
    fusion = next(n for n in graph["nodes"] if n["id"] == "fusion")
    assert fusion["planned"] is True and fusion["state"] == "unknown"


def test_graph_planned_role_reported_by_api_is_not_planned_anymore():
    graph = vm.build_graph([agent("fusion", "fusion")], [], vm.default_routes())
    fusion = next(n for n in graph["nodes"] if n["id"] == "fusion")
    assert fusion["planned"] is False and fusion["state"] == "ok"


def test_graph_unknown_role_reported_by_api_becomes_node():
    graph = vm.build_graph([agent("extra", "extra")], [], ROUTES)
    assert any(n["id"] == "extra" and n["state"] == "ok" for n in graph["nodes"])


def test_graph_edge_without_stats_has_no_rate():
    graph = vm.build_graph([], [], ROUTES)
    edge = next(e for e in graph["edges"] if e["topic"] == "vision.detections")
    assert edge["rate_per_s"] is None
    assert edge["label"] == "vision.detections"


def test_graph_nodes_have_layout_and_edges_have_kind():
    graph = vm.build_graph([agent("zeta", "zeta"), agent("eta", "eta")], [], ROUTES)
    known = next(n for n in graph["nodes"] if n["id"] == "ingest")
    assert (known["col"], known["row"]) == (0, 0)
    unknown = [n for n in graph["nodes"] if n["id"] in ("zeta", "eta")]
    assert len({(n["col"], n["row"]) for n in graph["nodes"]}) == len(graph["nodes"])
    assert all(n["row"] >= 3 for n in unknown)
    kinds = {e["topic"]: e["kind"] for e in vm.build_graph([], [], vm.default_routes())["edges"]}
    assert kinds["system.health"] == "control"
    assert kinds["vision.detections"] == "data"


def test_default_routes_cover_the_seven_roles():
    roles = {r.name for r in vm.default_routes()}
    assert {"ingest", "inference", "tracker", "event", "storage", "comms", "supervisor"} <= roles


# --- feed ---

def envelope(topic_name, payload=None, **extra):
    base = {
        "topic": topic_name, "source": "x", "payload": payload or {},
        "stream_id": None, "created_at": "2026-01-01T00:00:00+00:00",
        "event_id": "e1", "correlation_id": None, "causation_id": None,
    }
    base.update(extra)
    return base


def test_feed_entry_summarizes_and_keeps_correlation():
    entry = vm.feed_entry(envelope("events", {"outcome": "alert"}, correlation_id="c-9"))
    assert entry["topic"] == "events"
    assert entry["correlation_id"] == "c-9"
    assert entry["payload"] == {"outcome": "alert"}
    assert "outcome" in entry["summary"]


def test_filter_feed_by_topics_and_correlation_and_text():
    entries = [
        vm.feed_entry(envelope("events", {"a": 1}, correlation_id="c1", event_id="1")),
        vm.feed_entry(envelope("vision.tracks", {"a": 2}, event_id="2")),
        vm.feed_entry(envelope("events", {"zone_id": "bodega"}, event_id="3")),
    ]
    assert [e["event_id"] for e in vm.filter_feed(entries, topics={"events"})] == ["1", "3"]
    assert [e["event_id"] for e in vm.filter_feed(entries, correlation_id="c1")] == ["1"]
    assert [e["event_id"] for e in vm.filter_feed(entries, text="bodega")] == ["3"]
    assert len(vm.filter_feed(entries, limit=2)) == 2
    assert len(vm.filter_feed(entries)) == 3


def test_feed_redacts_biometric_keys():
    entry = vm.feed_entry(
        envelope("events", {"embedding": [0.1, 0.2], "nested": {"face_image": "AAAA"}, "ok": 1})
    )
    assert entry["payload"]["embedding"] == vm.REDACTED
    assert entry["payload"]["nested"]["face_image"] == vm.REDACTED
    assert entry["payload"]["ok"] == 1


# --- alertas ---

def decision_env(**over):
    payload = {
        "decision_id": "dec-1", "evaluated_at": "2026-01-01T00:00:00+00:00",
        "outcome": "alert", "confidence": 0.8,
        "reason_codes": ["person_without_tag", "stale_sensor"],
        "evidence": [{"evidence_id": "ev-1", "kind": "vision", "source": "tracker"}],
        "stream_id": "cam-1", "zone_id": "bodega", "person_id": "p-ab12",
        "requires_operator": True,
    }
    payload.update(over)
    return envelope("events", payload, source="fusion", correlation_id="c-1")


def test_format_alert_exposes_outcome_reasons_evidence_operator():
    alert = vm.format_alert(decision_env())
    assert alert["decision_id"] == "dec-1"
    assert alert["outcome"] == "alert"
    assert alert["severity"] == "high"
    assert alert["confidence_pct"] == 80
    assert [r["code"] for r in alert["reasons"]] == ["person_without_tag", "stale_sensor"]
    assert all(r["label"] for r in alert["reasons"])
    assert alert["evidence"][0]["evidence_id"] == "ev-1"
    assert alert["requires_operator"] is True
    assert alert["person_id"] == "p-ab12"
    assert alert["correlation_id"] == "c-1"


def test_format_alert_requires_operator_is_always_true():
    assert vm.format_alert(decision_env(requires_operator=False))["requires_operator"] is True


def test_format_alert_tolerates_missing_fields():
    alert = vm.format_alert(envelope("events", {"decision_id": "d", "outcome": "weird"}))
    assert alert["reasons"] == [] and alert["evidence"] == []
    assert alert["severity"] == "info"
    assert alert["confidence_pct"] is None


def test_format_alert_unknown_reason_code_kept_as_is():
    alert = vm.format_alert(decision_env(reason_codes=["nuevo_codigo"]))
    assert alert["reasons"][0] == {"code": "nuevo_codigo", "label": "nuevo_codigo"}


def test_format_alerts_keeps_order_and_limit():
    envs = [decision_env(decision_id=f"d{i}") for i in range(5)]
    alerts = vm.format_alerts(envs, limit=3)
    assert [a["decision_id"] for a in alerts] == ["d0", "d1", "d2"]


# --- mapa de sitio ---

SITE_TOML = """
[[cameras]]
id = "laptop-webcam"
[[receivers]]
id = "N0001"
kind = "pc-ble"
zone = "lobby"
[[zones]]
id = "lobby"
name = "Zona frente a la cámara"
receiver = "N0001"
camera = "laptop-webcam"
region = [0.05, 0.10, 0.95, 0.95]
enter_dbm = -70
calibrated = false
"""


def test_parse_site_map_reads_zones_receivers_cameras():
    site = vm.parse_site_map(SITE_TOML)
    assert site["zones"][0]["id"] == "lobby"
    assert site["zones"][0]["region"] == [0.05, 0.10, 0.95, 0.95]
    assert site["zones"][0]["calibrated"] is False
    assert site["receivers"][0] == {"id": "N0001", "kind": "pc-ble", "zone": "lobby"}
    assert site["cameras"][0]["id"] == "laptop-webcam"


def test_parse_site_map_invalid_raises_value_error_in_spanish():
    with pytest.raises(ValueError, match="mapa de sitio"):
        vm.parse_site_map("[[zones]\nid=")


def test_load_site_map_missing_file_returns_none(tmp_path):
    assert vm.load_site_map(tmp_path / "no.toml") is None
    assert vm.load_site_map(None) is None


def test_load_site_map_reads_file(tmp_path):
    path = tmp_path / "site.toml"
    path.write_text(SITE_TOML, encoding="utf-8")
    assert vm.load_site_map(path)["zones"][0]["id"] == "lobby"


def location(tag="tg-1", status="located", zone="lobby", confidence=0.9, **extra):
    payload = {"tag_ref": tag, "person_ref": "p-1", "status": status, "zone_id": zone,
               "confidence": confidence, "unknown_reason": None,
               "last_evidence_at": "2026-01-01T00:00:05+00:00"}
    payload.update(extra)
    return envelope("location", payload)


def test_is_location_by_topic_or_payload_shape():
    assert vm.is_location(location())
    assert vm.is_location(envelope("events", {"tag_ref": "t", "status": "located"}))
    assert not vm.is_location(envelope("events", {"decision_id": "d"}))


def test_tag_presence_uses_newest_message_per_tag():
    # los mensajes llegan del más nuevo al más antiguo
    msgs = [
        location("tg-1", "unknown", None, 0.0, unknown_reason="tag_missing"),
        location("tg-2", "unknown", None, 0.0, unknown_reason="stale_evidence"),
        location("tg-1", "located", "lobby", 0.9),
    ]
    by = {t["tag_ref"]: t for t in vm.tag_presence(msgs)}
    assert by["tg-1"]["inside"] is False
    assert by["tg-1"]["unknown_reason"] == "tag_missing"
    assert by["tg-2"]["unknown_reason"] == "stale_evidence"


def test_site_view_places_tags_in_their_zone():
    view = vm.site_view(vm.parse_site_map(SITE_TOML), [location("tg-1", "located", "lobby", 0.9)])
    assert view["has_location_data"] is True
    tag = view["zones"][0]["tags"][0]
    assert tag["tag_ref"] == "tg-1" and tag["confidence_pct"] == 90
    assert view["zones"][0]["receiver"]["id"] == "N0001"


def test_site_view_without_location_data_says_no_data():
    view = vm.site_view(vm.parse_site_map(SITE_TOML), [])
    assert view["has_location_data"] is False
    assert view["zones"][0]["tags"] == []


def test_site_view_without_site_map_is_none():
    assert vm.site_view(None, []) is None


# --- salud ---

def test_health_strip_combines_api_and_streams():
    api = {"status": "degraded", "agents_total": 7, "agents_running": 6,
           "agents_failed": 1, "uptime_s": 12.0, "ws_clients": 2, "ws_dropped_total": 5}
    streams = vm.stream_health(
        [
            # del más nuevo al más antiguo
            envelope("system.health", {"stream_id": "cam-1", "state": "reconnecting"}),
            envelope("stream.status", {"state": "up"}, stream_id="cam-2"),
            envelope("system.health", {"stream_id": "cam-1", "state": "connected"}),
        ]
    )
    by = {s["stream_id"]: s for s in streams}
    assert by["cam-1"]["state"] == "reconnecting"
    assert by["cam-2"]["state"] == "up"
    strip = vm.health_strip(api, streams, connected=True)
    assert strip["status"] == "degraded"
    assert strip["api_connected"] is True
    assert strip["hardware"] == "laptop"
    assert strip["agents_failed"] == 1


def test_stream_health_keeps_metrics_when_present():
    streams = vm.stream_health(
        [envelope("system.health", {"stream_id": "cam-1", "state": "connected", "fps": 24.5,
                                    "latency_ms_p50": 11.0, "latency_ms_p90": 20.0})]
    )
    assert streams[0]["fps"] == 24.5 and streams[0]["latency_p90_ms"] == 20.0
    assert streams[0]["latency_p50_ms"] == 11.0


def test_health_strip_without_api_is_offline():
    strip = vm.health_strip(None, [], connected=False)
    assert strip["status"] == "offline"
    assert strip["api_connected"] is False


@pytest.mark.parametrize(
    "outcome", ["corroborated", "alert", "inconclusive", "not_restricted", "uncorroborated"]
)
def test_every_outcome_has_spanish_label_and_keeps_raw_value(outcome):
    alert = vm.format_alert(decision_env(outcome=outcome))
    assert alert["outcome_raw"] == outcome
    assert alert["outcome_label"] != outcome
    assert alert["outcome_label"][0].isupper()


def test_unknown_outcome_label_is_spanish_and_raw_kept():
    alert = vm.format_alert(decision_env(outcome="weird"))
    assert alert["outcome_label"] == "Resultado desconocido"
    assert alert["outcome_raw"] == "weird"


def test_evidence_entries_get_spanish_kind_and_role_labels():
    alert = vm.format_alert(
        decision_env(evidence=[{"evidence_id": "e", "kind": "location", "role": "supports"}])
    )
    ev = alert["evidence"][0]
    assert ev["kind_label"] == "Localización" and ev["role_label"] == "Respalda"
    odd = vm.format_alert(decision_env(evidence=[{"evidence_id": "e", "kind": "x"}]))
    assert odd["evidence"][0]["kind_label"] == "x"
