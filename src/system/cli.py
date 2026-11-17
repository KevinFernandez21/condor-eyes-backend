"""CLI del ejecutor: ``scripts/run_system.py --profile sim|laptop|replay``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from comms.api import ConfigurationError, check_bind

from .app import SystemApp
from .config import DEFAULT_CONFIG_PATH, ConfigError, load_system_config

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_ERROR = 1


def _positive(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"no es un número: {text!r}") from None
    if not value > 0 or value == float("inf"):
        raise argparse.ArgumentTypeError(
            "la duración debe ser un número positivo finito"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_system",
        description="Ejecuta todo el sistema multiagente de Condor Eye en este equipo.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="sim | laptop | replay (por defecto el de configs/system.toml)",
    )
    parser.add_argument(
        "--duration",
        type=_positive,
        default=None,
        help="segundos a ejecutar; sin valor corre hasta Ctrl+C",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="sirve la API de observabilidad (solo lectura) de #42",
    )
    parser.add_argument("--api-host", default=None, help="por defecto el de [api].host")
    parser.add_argument(
        "--api-port", type=int, default=None, help="por defecto el de [api].port"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="token de la API (o variable CONDOR_API_TOKEN); obligatorio fuera de loopback",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="semilla de la simulación (escenas, caídas, seudónimos); aleatoria por defecto",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--verbose", action="store_true", help="logs DEBUG")
    parser.add_argument(
        "--debug", action="store_true", help="muestra el traceback ante errores"
    )
    return parser


def format_summary(summary: dict[str, Any]) -> str:
    """Resumen legible: agentes, mensajes por tópico y estado de componentes."""
    lines = [f"perfil={summary['profile']} uptime={summary['uptime_s']}s"]
    lines.append("agentes:")
    for agent in summary["agents"]:
        if agent["role"] == "component":
            continue
        lines.append(
            f"  {agent['name']:<12} {agent['state']:<8} procesados={agent['processed']} "
            f"fallos={agent['failures']}"
        )
    lines.append("mensajes por topic:")
    for stat in summary["topics"]:
        lines.append(f"  {stat['topic']:<20} {stat['count']}")
    lines.append("componentes:")
    for name, comp in summary["components"].items():
        plugin = f" plugin={comp['plugin']}" if "plugin" in comp else ""
        lines.append(
            f"  {name:<10} {comp['status']:<10}{plugin} {comp.get('detail', '')}"
        )
    return "\n".join(lines)


def config_is_sim(app: SystemApp) -> bool:
    return app.config.camera.kind == "fake"


async def _run(app: SystemApp, duration: float | None) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def request_stop(*_: Any) -> None:
        loop.call_soon_threadsafe(stop.set)

    previous: dict[int, Any] = {}
    signals: list[int] = [signal.SIGINT]
    if hasattr(signal, "SIGBREAK"):  # Ctrl+Break en Windows
        signals.append(signal.SIGBREAK)
    elif hasattr(signal, "SIGTERM"):
        signals.append(signal.SIGTERM)
    for sig in signals:
        previous[sig] = signal.signal(sig, request_stop)
    last: dict[str, Any] = {}
    try:
        await app.start()
        print(
            f"Sistema en marcha (perfil {app.config.profile}). Ctrl+C para detener.",
            flush=True,
        )
        if config_is_sim(app):
            print(f"Semilla de la simulación: {app.seed}", flush=True)
        if app.api_url:
            print(f"API de observabilidad: {app.api_url}", flush=True)
        waiter = asyncio.ensure_future(stop.wait())
        try:
            await asyncio.wait({waiter}, timeout=duration)
        finally:
            waiter.cancel()
        print("Deteniendo...", flush=True)
        last = {
            **app.snapshot(),
            "agents": app.view.agents(),
            "topics": app.view.topics(),
        }
    finally:
        await app.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return last


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_system_config(args.config, args.profile)
    except ConfigError as exc:
        print(f"Error de configuración (perfil/archivo): {exc}", file=sys.stderr)
        return EXIT_CONFIG
    token = args.token or os.environ.get("CONDOR_API_TOKEN") or None
    if args.token and not args.api:
        print("--token solo tiene sentido con --api.", file=sys.stderr)
        return EXIT_CONFIG
    if args.api:
        api = replace(
            config.api,
            enabled=True,
            host=args.api_host or config.api.host,
            port=config.api.port if args.api_port is None else args.api_port,
        )
        config = replace(config, api=api)
        try:
            check_bind(api.host, token)
        except ConfigurationError as exc:
            print(f"Error de configuración de la API: {exc}", file=sys.stderr)
            return EXIT_CONFIG
    app = SystemApp(config, api_token=token, seed=args.seed)
    try:
        snapshot = asyncio.run(_run(app, args.duration))
    except Exception as exc:
        if args.debug:
            raise
        print(
            f"Error al ejecutar el sistema ({type(exc).__name__}): {exc}. "
            "Use --debug para ver el traceback.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    if snapshot:
        print(format_summary(snapshot), flush=True)
    return EXIT_OK
