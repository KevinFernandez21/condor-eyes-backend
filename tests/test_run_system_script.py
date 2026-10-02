"""CLI scripts/run_system.py: salida, códigos de error y apagado con SIGINT."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from system.cli import main

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_system.py"


def run_script(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=ROOT,
    )


def test_sim_duracion_imprime_resumen_y_sale_con_cero():
    done = run_script("--profile", "sim", "--duration", "1.5")
    assert done.returncode == 0, done.stderr
    assert "perfil=sim" in done.stdout
    assert "vision.detections" in done.stdout
    for role in (
        "ingest",
        "inference",
        "tracker",
        "event",
        "storage",
        "comms",
        "supervisor",
    ):
        assert role in done.stdout


def test_perfil_invalido_sale_con_2():
    done = run_script("--profile", "jetson", "--duration", "1")
    assert done.returncode == 2
    assert "perfil" in done.stderr.lower()


def test_config_inexistente_sale_con_2(tmp_path):
    done = run_script("--config", str(tmp_path / "x.toml"), "--duration", "1")
    assert done.returncode == 2
    assert "No se encontró" in done.stderr


def test_duracion_negativa_se_rechaza():
    done = run_script("--duration", "-3")
    assert done.returncode == 2


def test_main_en_proceso_devuelve_cero(capsys):
    assert main(["--profile", "sim", "--duration", "0.5"]) == 0
    assert "perfil=sim" in capsys.readouterr().out


def test_sigint_apaga_limpio(capsys):
    """Ctrl+C real (señal SIGINT al proceso) cierra todo y devuelve 0."""
    timer = threading.Timer(1.0, lambda: signal.raise_signal(signal.SIGINT))
    timer.start()
    try:
        code = main(["--profile", "sim"])
    finally:
        timer.cancel()
    out = capsys.readouterr().out
    assert code == 0
    assert "Deteniendo" in out
    leaked = [t.name for t in threading.enumerate() if t.name.startswith("video")]
    assert leaked == []


@pytest.mark.skipif(sys.platform != "win32", reason="CTRL_BREAK es de Windows")
def test_ctrl_break_en_subproceso_windows():
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--profile", "sim"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        import time

        time.sleep(4.0)
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        out, err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    assert "Deteniendo" in out


def test_error_inesperado_no_muestra_traceback(monkeypatch, capsys):
    from system import cli

    async def boom(self):
        raise RuntimeError("fallo de prueba")

    monkeypatch.setattr(cli.SystemApp, "start", boom)
    assert main(["--profile", "sim", "--duration", "1"]) == 1
    err = capsys.readouterr().err
    assert "fallo de prueba" in err
    assert "--debug" in err
    assert "Traceback" not in err


def test_debug_deja_pasar_la_excepcion(monkeypatch):
    from system import cli

    async def boom(self):
        raise RuntimeError("fallo de prueba")

    monkeypatch.setattr(cli.SystemApp, "start", boom)
    with pytest.raises(RuntimeError, match="fallo de prueba"):
        main(["--profile", "sim", "--duration", "1", "--debug"])


def test_api_imprime_la_url(capsys):
    assert (
        main(["--profile", "sim", "--api", "--api-port", "0", "--duration", "1"]) == 0
    )
    out = capsys.readouterr().out
    assert "API de observabilidad: http://127.0.0.1:" in out


def test_api_host_publico_sin_token_sale_con_2(capsys):
    code = main(["--api", "--api-host", "0.0.0.0", "--duration", "1"])
    assert code == 2
    assert "token" in capsys.readouterr().err.lower()


def test_api_host_publico_con_token_arranca(capsys):
    code = main(
        [
            "--api",
            "--api-host",
            "0.0.0.0",
            "--api-port",
            "0",
            "--token",
            "t",
            "--duration",
            "1",
        ]
    )
    assert code == 0
    assert "API de observabilidad:" in capsys.readouterr().out


def test_token_sin_api_se_rechaza(capsys):
    assert main(["--token", "x", "--duration", "1"]) == 2


def test_api_con_puerto_ocupado_sale_con_1_y_mensaje(capsys):
    import socket

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    try:
        port = str(blocker.getsockname()[1])
        code = main(["--api", "--api-port", port, "--duration", "1"])
    finally:
        blocker.close()
    assert code == 1
    assert f"puerto {port}" in capsys.readouterr().err
