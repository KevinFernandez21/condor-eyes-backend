"""Utilidades compartidas por las pruebas del ejecutor del sistema."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from pathlib import Path

from system.config import SystemConfig, load_system_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "system.toml"


def fast_config(profile: str = "sim", **overrides) -> SystemConfig:
    """Config real del repo con intervalos de milisegundos para pruebas rápidas."""
    cfg = load_system_config(CONFIG_PATH, profile)
    cfg = dataclasses.replace(
        cfg,
        heartbeat_interval_s=0.1,
        fusion_interval_s=0.1,
        shutdown_timeout_s=3.0,
        camera=dataclasses.replace(cfg.camera, fps=50.0),
        cameras=tuple(dataclasses.replace(c, fps=50.0) for c in cfg.cameras),
        scenes=dataclasses.replace(cfg.scenes, dropouts=False),
        tag=dataclasses.replace(cfg.tag, interval_s=0.05),
        identity=dataclasses.replace(cfg.identity, interval_s=0.05),
        actuator=dataclasses.replace(cfg.actuator, interval_s=0.05),
    )
    return dataclasses.replace(cfg, **overrides)


def thread_names() -> set[str]:
    return {t.name for t in threading.enumerate()}


def other_tasks() -> set[asyncio.Task]:
    current = asyncio.current_task()
    return {t for t in asyncio.all_tasks() if t is not current and not t.done()}
