"""Resource and user-data paths that work from source and from a PyInstaller bundle."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_root() -> Path:
    """Read-only bundled resources (mappings, sample data, model weights)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Writable per-user directory for the database, logs and exports.

    Overridable with AUTOGUARD_HOME (used by tests)."""
    override = os.environ.get("AUTOGUARD_HOME")
    if override:
        p = Path(override)
    else:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
        p = Path(base) / "AutoGuardAI"
    p.mkdir(parents=True, exist_ok=True)
    return p


MAPPINGS_DIR = resource_root() / "datasets" / "column_mappings"
SAMPLE_DIR = resource_root() / "data" / "sample"
MODELS_DIR = resource_root() / "models"
