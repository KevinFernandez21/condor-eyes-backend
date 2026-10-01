"""Punto de entrada del CLI de armas: `uv run python scripts/firearm.py --help`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firearm.cli import main

if __name__ == "__main__":
    # La consola de Windows usa cp1252 por defecto; los mensajes van en español.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    main()
