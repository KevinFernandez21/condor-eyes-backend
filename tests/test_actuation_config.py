"""Pruebas de la configuración TOML del nodo Pan-Tilt."""

from pathlib import Path

import pytest

from actuation.config import ActuationConfig, load_actuation_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pan_tilt.toml"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pt.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repo_config_loads_and_is_consistent():
    cfg = load_actuation_config(REPO_CONFIG)
    assert isinstance(cfg, ActuationConfig)
    limits = cfg.limits
    assert limits.pan_min_deg < limits.neutral_pan_deg < limits.pan_max_deg
    assert limits.tilt_min_deg < limits.neutral_tilt_deg < limits.tilt_max_deg
    assert cfg.control.max_speed_dps <= limits.max_speed_dps
    assert cfg.comms.node_watchdog_s > cfg.comms.heartbeat_interval_s


def test_defaults_without_file():
    cfg = load_actuation_config(None)
    assert cfg.limits.max_speed_dps > 0


def test_unknown_key_is_rejected(tmp_path):
    path = _write(tmp_path, "[pan_tilt.limits]\nfoo = 1\n")
    with pytest.raises(ValueError, match="desconocidas"):
        load_actuation_config(path)


def test_unknown_section_is_rejected(tmp_path):
    path = _write(tmp_path, "[pan_tilt.bar]\nx = 1\n")
    with pytest.raises(ValueError, match="desconocida"):
        load_actuation_config(path)


@pytest.mark.parametrize(
    "text",
    [
        "[pan_tilt.limits]\npan_min_deg = 10.0\npan_max_deg = -10.0\n",
        "[pan_tilt.limits]\nneutral_pan_deg = 200.0\n",
        "[pan_tilt.limits]\nmax_speed_dps = 0.0\n",
        "[pan_tilt.limits]\nmax_speed_dps = 30.0\n[pan_tilt.control]\nmax_speed_dps = 50.0\n",
        "[pan_tilt.control]\nkp = 0.0\n",
        "[pan_tilt.control]\ndeadband = 1.0\n",
        "[pan_tilt.control]\nsmoothing_alpha = 0.0\n",
        "[pan_tilt.comms]\nnode_watchdog_s = 0.1\nheartbeat_interval_s = 0.5\n",
        "[pan_tilt.comms]\nack_timeout_s = 2.0\ncomms_timeout_s = 1.0\n",
        "[pan_tilt.simulator]\ndrop_prob = 1.5\n",
    ],
)
def test_invalid_values_are_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        load_actuation_config(_write(tmp_path, text))
