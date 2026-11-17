"""Arranca todo el sistema multiagente en este equipo (sim | laptop | replay)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from system.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
