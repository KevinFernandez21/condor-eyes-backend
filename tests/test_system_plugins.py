"""Registro de plugins opcionales descubiertos por importación."""

from __future__ import annotations

import json

from system.plugins import KNOWN_PLUGINS, PluginRegistry


def finder_with(*present: str):
    return lambda module: module in present


def test_los_cuatro_plugins_opcionales_estan_declarados():
    assert {"location", "identity", "actuation", "tagbridge"} <= set(KNOWN_PLUGINS)


def test_ausente_es_not_installed():
    registry = PluginRegistry(finder=finder_with())
    infos = registry.discover()
    assert all(info.installed is False for info in infos.values())
    health = registry.health("location")
    assert health["status"] == "not_installed"
    assert "location" in health["detail"]


def test_presente_se_marca_instalado():
    registry = PluginRegistry(finder=finder_with("tagbridge"))
    assert registry.discover()["tagbridge"].installed is True
    assert registry.discover()["location"].installed is False
    assert registry.health("tagbridge")["status"] == "installed"


def test_finder_que_lanza_cuenta_como_ausente():
    def boom(module: str) -> bool:
        raise ModuleNotFoundError(module)

    registry = PluginRegistry(finder=boom)
    assert registry.discover()["actuation"].installed is False


def test_find_spec_real_con_submodulo_inexistente_no_lanza():
    registry = PluginRegistry()  # find_spec real
    # faceid existe en main pero faceid.identity todavía no: no debe lanzar.
    assert isinstance(registry.discover()["identity"].installed, bool)


def test_salud_es_json_estricto():
    registry = PluginRegistry(finder=finder_with())
    json.dumps(registry.health("identity"), allow_nan=False)
