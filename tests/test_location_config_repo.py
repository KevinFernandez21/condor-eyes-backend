"""Pruebas de configuración, seudonimización y repositorio de personal."""

import logging
from pathlib import Path

import pytest

from location.config import ConfigError, LocationConfig, load_config, parse_config
from location.privacy import Pseudonymizer
from location.repository import (
    AccessDenied,
    InMemoryPersonnelRepository,
    PersonnelRecord,
    Principal,
)

REPO_CONFIG = Path(__file__).parent.parent / "configs" / "location.toml"

ZONES = {
    "lobby": {"nodes": ["N0001"], "neighbors": ["hall"]},
    "hall": {"nodes": ["N0002", "N0003"], "neighbors": ["lobby", "lab"]},
    "lab": {"nodes": ["N0004"], "neighbors": ["hall"]},
}


def test_repo_config_file_loads_and_is_consistent():
    cfg = load_config(REPO_CONFIG)
    assert isinstance(cfg, LocationConfig)
    assert cfg.zone_of_node("N0001") is not None
    assert cfg.smoothing_window_s <= cfg.evidence_max_age_s


def test_parse_config_defaults_and_zone_lookup():
    cfg = parse_config({"zones": ZONES})
    assert cfg.zone_of_node("N0003") == "hall"
    assert cfg.zone_of_node("N9999") is None
    assert cfg.are_adjacent("lobby", "hall")
    assert cfg.are_adjacent("hall", "lobby")  # simétrica aunque se declare una vez
    assert not cfg.are_adjacent("lobby", "lab")
    assert cfg.are_adjacent("lab", "lab")


def test_parse_config_overrides():
    cfg = parse_config({"smoothing": {"window_s": 4, "alpha": 0.8}, "zones": ZONES})
    assert cfg.smoothing_window_s == 4
    assert cfg.smoothing_alpha == 0.8


@pytest.mark.parametrize(
    "raw, match",
    [
        ({}, "zonas"),
        ({"zones": {"a": {"nodes": []}}}, "nodo"),
        (
            {"zones": {"a": {"nodes": ["N1"]}, "b": {"nodes": ["N1"]}}},
            "más de una zona",
        ),
        ({"zones": {"a": {"nodes": ["N1"], "neighbors": ["x"]}}}, "vecina"),
        ({"smoothing": {"alpha": 0}, "zones": ZONES}, "alpha"),
        (
            {
                "smoothing": {"window_s": 99},
                "freshness": {"evidence_max_age_s": 5},
                "zones": ZONES,
            },
            "window_s",
        ),
        ({"freshness": {"max_age_s": -1}, "zones": ZONES}, "max_age_s"),
        (
            {
                "confidence": {"rssi_floor_dbm": -50, "rssi_strong_dbm": -60},
                "zones": ZONES,
            },
            "rssi",
        ),
    ],
)
def test_parse_config_rejects_invalid(raw, match):
    with pytest.raises(ConfigError, match=match):
        parse_config(raw)


def test_pseudonym_is_stable_keyed_and_opaque():
    p1 = Pseudonymizer(b"clave-1")
    p2 = Pseudonymizer(b"clave-2")
    assert p1.ref("tag", "A1B2C3D4E5F6") == p1.ref("tag", "A1B2C3D4E5F6")
    assert p1.ref("tag", "A1B2C3D4E5F6") != p2.ref("tag", "A1B2C3D4E5F6")
    assert p1.ref("tag", "A1B2C3D4E5F6") != p1.ref("person", "A1B2C3D4E5F6")
    assert "A1B2C3D4E5F6" not in p1.ref("tag", "A1B2C3D4E5F6")
    assert p1.ref("tag", "A1B2C3D4E5F6").startswith("tag-")


def test_pseudonymizer_requires_a_non_empty_key():
    with pytest.raises(ValueError, match="clave"):
        Pseudonymizer(b"")


def test_pseudonymizer_random_key_differs_between_instances():
    assert Pseudonymizer.random().ref("tag", "X") != Pseudonymizer.random().ref(
        "tag", "X"
    )


def make_repo():
    repo = InMemoryPersonnelRepository(Pseudonymizer(b"k"))
    repo.enroll(
        "A1B2C3D4E5F6",
        PersonnelRecord(
            person_id="P-1001",
            display_name="Ana Pérez",
            allowed_zones=frozenset({"lobby", "hall"}),
        ),
    )
    return repo


READER = Principal("localization", frozenset({"personnel:read"}))


def test_repository_lookup_requires_scope():
    repo = make_repo()
    record = repo.lookup("A1B2C3D4E5F6", requester=READER)
    assert record is not None and record.person_id == "P-1001"
    with pytest.raises(AccessDenied):
        repo.lookup("A1B2C3D4E5F6", requester=Principal("intruso", frozenset()))
    with pytest.raises(AccessDenied):
        repo.enrolled_tags(requester=Principal("intruso", frozenset({"x"})))


def test_repository_unknown_tag_returns_none_and_lists_enrolled():
    repo = make_repo()
    assert repo.lookup("FFFFFFFFFFFF", requester=READER) is None
    assert repo.enrolled_tags(requester=READER) == frozenset({"A1B2C3D4E5F6"})


def test_record_repr_hides_personal_data():
    text = repr(make_repo().lookup("A1B2C3D4E5F6", requester=READER))
    assert "Ana" not in text and "P-1001" not in text


def test_repository_never_logs_plaintext(caplog):
    repo = make_repo()
    with caplog.at_level(logging.DEBUG):
        repo.lookup("A1B2C3D4E5F6", requester=READER)
        repo.lookup("FFFFFFFFFFFF", requester=READER)
        with pytest.raises(AccessDenied):
            repo.lookup("A1B2C3D4E5F6", requester=Principal("intruso", frozenset()))
    assert caplog.records, "se esperaba auditoría de los accesos"
    text = caplog.text
    for secret in ("A1B2C3D4E5F6", "FFFFFFFFFFFF", "P-1001", "Ana"):
        assert secret not in text
    assert "intruso" in text  # el solicitante sí se audita
