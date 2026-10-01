"""Punto de entrada del detector de vigilancia: `uv run python scripts/surveillance.py --help`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surveillance.cli import main

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    main()
