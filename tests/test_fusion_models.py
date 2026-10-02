"""Tests de la configuración y los tipos de entrada de la fusión de evidencia."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fusion import (
    FusionConfig,
    IdentityEvidence,
    IdentityStatus,
    InMemoryPermissions,
    LocationEvidence,
    TrackObservation,
    load_fusion_config,
)

T0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
CONFIG_TOML = Path(__file__).resolve().parents[1] / "configs" / "fusion.toml"


def test_default_toml_loads_and_matches_dataclass_defaults():
    assert load_fusion_config(CONFIG_TOML) == FusionConfig()


def test_unknown_key_in_toml_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[fusion]\nmax_edad_pista = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Claves desconocidas"):
        load_fusion_config(bad)


def test_override_and_range_validation():
    cfg = load_fusion_config(None, min_identity_score=0.9)
    assert cfg.min_identity_score == 0.9
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_fusion_config(None, min_identity_score=1.5)
    with pytest.raises(ValueError, match="positivo"):
        load_fusion_config(None, location_max_age_s=0)


def test_naive_datetimes_are_rejected():
    naive = T0.replace(tzinfo=None)
    with pytest.raises(ValueError, match="zona horaria"):
        TrackObservation("e1", "cam1/1", "cam1", "z1", naive, 0.9)


def test_confidence_out_of_range_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        LocationEvidence("l1", "p1", "z1", T0, 1.2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        IdentityEvidence("i1", "cam1/1", IdentityStatus.MATCH, T0, "p1", -0.1)


def test_permissions_repository_distinguishes_missing_record():
    repo = InMemoryPermissions({"p1": frozenset({"z1"})})
    assert repo.allowed_zones("p1") == frozenset({"z1"})
    assert repo.allowed_zones("ghost") is None
