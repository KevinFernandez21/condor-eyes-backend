"""Carga y validación estricta de configs/system.toml."""

from __future__ import annotations

from pathlib import Path

import pytest
from system_support import CONFIG_PATH

from system.config import ConfigError, load_system_config


@pytest.mark.parametrize("profile", ["sim", "laptop", "replay"])
def test_cada_perfil_del_repo_carga(profile):
    cfg = load_system_config(CONFIG_PATH, profile)
    assert cfg.profile == profile
    assert cfg.zones, "debe definir zonas"
    assert cfg.heartbeat_interval_s > 0


def test_perfil_por_defecto_es_sim():
    assert load_system_config(CONFIG_PATH).profile == "sim"


def test_sim_no_necesita_hardware():
    cfg = load_system_config(CONFIG_PATH, "sim")
    assert (cfg.camera.kind, cfg.tag.kind, cfg.actuator.kind) == ("fake", "sim", "sim")


def test_laptop_usa_webcam_y_c6():
    cfg = load_system_config(CONFIG_PATH, "laptop")
    assert cfg.camera.kind == "webcam"
    assert cfg.tag.kind == "c6"
    assert cfg.actuator.kind == "sim"


def test_replay_usa_video_y_jsonl():
    cfg = load_system_config(CONFIG_PATH, "replay")
    assert cfg.camera.kind == "file"
    assert cfg.tag.kind == "replay"


def test_perfil_desconocido():
    with pytest.raises(ConfigError, match="Perfil desconocido"):
        load_system_config(CONFIG_PATH, "jetson")


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "system.toml"
    path.write_text(body, encoding="utf-8")
    return path


BASE = """
[system]
default_profile = "sim"
{system}
[[zones]]
zone_id = "a"
restricted = false
x_min = 0.0
x_max = 1.0
[profiles.sim.camera]
kind = "fake"
{camera}
"""


def _cfg(tmp_path, system="", camera=""):
    return _write(tmp_path, BASE.format(system=system, camera=camera))


def test_clave_desconocida_en_system(tmp_path):
    with pytest.raises(ConfigError, match=r"\[system\].*bogus"):
        load_system_config(_cfg(tmp_path, system="bogus = 1"), "sim")


def test_clave_desconocida_en_perfil(tmp_path):
    with pytest.raises(ConfigError, match="camera.*bogus"):
        load_system_config(_cfg(tmp_path, camera="bogus = 1"), "sim")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_numeros_no_finitos_se_rechazan(tmp_path, value):
    path = _cfg(tmp_path, system=f"heartbeat_interval_s = {value}")
    with pytest.raises(ConfigError, match="finito"):
        load_system_config(path, "sim")


def test_bool_no_es_numero(tmp_path):
    with pytest.raises(ConfigError, match="numérico"):
        load_system_config(_cfg(tmp_path, system="heartbeat_interval_s = true"), "sim")


def test_numero_no_positivo(tmp_path):
    with pytest.raises(ConfigError, match="positivo"):
        load_system_config(_cfg(tmp_path, system="heartbeat_interval_s = 0"), "sim")


def test_kind_invalido(tmp_path):
    body = BASE.format(system="", camera="").replace('kind = "fake"', 'kind = "ptz"')
    with pytest.raises(ConfigError, match=r"camera\]\.kind inv"):
        load_system_config(_write(tmp_path, body), "sim")


def test_zona_con_rango_invertido(tmp_path):
    body = BASE.format(system="", camera="").replace("x_max = 1.0", "x_max = -1.0")
    with pytest.raises(ConfigError, match="x_min"):
        load_system_config(_write(tmp_path, body), "sim")


def test_zonas_duplicadas(tmp_path):
    body = BASE.format(system="", camera="") + (
        '[[zones]]\nzone_id = "a"\nrestricted = true\nx_min = 0.0\nx_max = 1.0\n'
    )
    with pytest.raises(ConfigError, match="duplicada"):
        load_system_config(_write(tmp_path, body), "sim")


def test_archivo_inexistente(tmp_path):
    with pytest.raises(ConfigError, match="No se encontró"):
        load_system_config(tmp_path / "nada.toml", "sim")


def test_toml_invalido(tmp_path):
    with pytest.raises(ConfigError, match="TOML"):
        load_system_config(_write(tmp_path, "esto no es toml ["), "sim")


def test_permisos_deben_ser_listas_de_texto(tmp_path):
    body = BASE.format(system="", camera="") + "[permissions]\np1 = [1, 2]\n"
    with pytest.raises(ConfigError, match="permissions"):
        load_system_config(_write(tmp_path, body), "sim")
