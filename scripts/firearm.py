"""Punto de entrada del CLI de armas: `uv run python scripts/firearm.py --help`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firearm.cli import main

if __name__ == "__main__":
    main()
