"""Hash-checked persistence for joblib artifacts produced by this project.

joblib unpickles code, so we only ever load artifacts created by our own
training scripts, and only if their SHA-256 matches the hash recorded in the
accompanying JSON at training time."""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from core.data_sources.files import sha256_file


def save_artifact(obj, meta: dict, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, directory / f"{stem}.joblib")
    meta = {**meta, "artifact_sha256": sha256_file(directory / f"{stem}.joblib")}
    (directory / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_artifact(directory: Path, stem: str) -> tuple[object, dict]:
    meta = json.loads((directory / f"{stem}_meta.json").read_text(encoding="utf-8"))
    if sha256_file(directory / f"{stem}.joblib") != meta.get("artifact_sha256"):
        raise ValueError(f"{stem}.joblib hash does not match {stem}_meta.json - refusing to load.")
    return joblib.load(directory / f"{stem}.joblib"), meta
