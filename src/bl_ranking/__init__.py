"""bl_ranking -- production wrapper around the vendored BL research scripts.

Importing this package installs the compact name index in place of
``names_dataset`` (see ``name_index``). It has to happen here, before anything
imports ``bl_ranking.original.*``: both vendored scripts build
``NameDataset()`` at import time, costing ~1.9 GB and ~6 s per process.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

__all__ = ["REPO_ROOT", "bootstrap_name_index"]

REPO_ROOT = Path(__file__).resolve().parents[2]

_logger = logging.getLogger("bl_ranking")


def _resolve(path_str: str) -> Path:
    """Resolve a configured path against the CWD first, then the repo root,
    so the same default works from a shell in any directory and from inside
    a container whose WORKDIR differs."""
    candidate = Path(path_str)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPO_ROOT / candidate


def bootstrap_name_index() -> bool:
    """Install the compact name index. Returns True if it is active.

    Reads the path straight from the environment rather than through
    ``Settings`` to keep this import as light as it is early; ``config.py``
    owns the same default and is the documented place to change it.
    """
    if os.environ.get("BL_DISABLE_NAME_INDEX") == "1":
        _logger.warning("BL_DISABLE_NAME_INDEX=1 — using the real names_dataset (~1.9 GB RSS)")
        return False

    from bl_ranking.name_index import install_shim

    path = _resolve(os.environ.get("NAME_INDEX_PATH", "assets/name_gender_index.json.gz"))
    required = os.environ.get("BL_REQUIRE_NAME_INDEX") == "1"
    return install_shim(path, required=required)


bootstrap_name_index()
