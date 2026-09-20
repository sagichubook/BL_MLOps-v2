#!/usr/bin/env python3
"""Thin CLI shim — see bl_ranking.serve_cli."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bl_ranking.serve_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
