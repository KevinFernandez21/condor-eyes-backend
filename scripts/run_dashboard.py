"""Dashboard del sistema multiagente (#43).

Uso (desde la raíz del repo)::

    # contra una API de observabilidad ya en marcha
    CONDOR_API_URL=http://127.0.0.1:8000 uv run python scripts/run_dashboard.py

    # vistazo rápido: levanta también la demo sintética de la API (#42)
    uv run python scripts/run_dashboard.py --demo

Abre http://127.0.0.1:8080. Variables: ``CONDOR_API_URL``, ``CONDOR_API_TOKEN``,
``CONDOR_SITE_MAP`` (opcional; por defecto ``configs/site_map.toml`` si existe).
El dashboard solo es cliente de la API: nunca toca el bus ni el runtime.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dashboard import (
    DashboardConfig,
    DashboardConfigurationError,
    create_app,
)
from dashboard.server import check_bind


def _wait_for_api(url: str, token: str | None, timeout: float) -> bool:
    request = urllib.request.Request(f"{url}/health")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=1) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="dónde sirve el dashboard")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--allow-lan",
        action="store_true",
        help="permite escuchar fuera de loopback (el dashboard no tiene login propio)",
    )
    parser.add_argument("--api-url", default=None, help="sobrescribe CONDOR_API_URL")
    parser.add_argument("--site-map", default=None, help="sobrescribe CONDOR_SITE_MAP")
    parser.add_argument(
        "--demo", action="store_true", help="arranca también scripts/run_comms_demo.py"
    )
    parser.add_argument("--demo-port", type=int, default=8000)
    args = parser.parse_args()

    config = DashboardConfig.from_env()
    if args.api_url:
        config = dataclasses.replace(config, api_url=args.api_url.rstrip("/"))
    if args.site_map:
        config = dataclasses.replace(config, site_map_path=args.site_map)
    try:
        check_bind(args.host, args.allow_lan)
    except DashboardConfigurationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    demo: subprocess.Popen[bytes] | None = None
    if args.demo:
        config = dataclasses.replace(config, api_url=f"http://127.0.0.1:{args.demo_port}")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_comms_demo.py"),
            "--port",
            str(args.demo_port),
        ]
        if config.token:
            command += ["--token", config.token]
        demo = subprocess.Popen(command)
        if not _wait_for_api(config.api_url, config.token, timeout=20):
            print("Error: la demo de la API no respondió a tiempo.", file=sys.stderr)
            demo.terminate()
            return 1

    import uvicorn  # diferido: importar el paquete no arranca nada

    app = create_app(config)
    print(f"Dashboard en http://{args.host}:{args.port}  (API: {config.api_url})", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        if demo is not None:
            demo.terminate()
            try:
                demo.wait(timeout=5)
            except subprocess.TimeoutExpired:
                demo.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
