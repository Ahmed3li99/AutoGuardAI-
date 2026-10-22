"""SETUP-TIME ONLY: download MOMENT-1-small weights at a pinned revision into
models/moment-1-small/ and record file hashes. The application itself never
downloads anything; it loads this directory with offline mode forced on.

Usage:  python tools/download_moment.py [--revision <commit-sha>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "AutonLab/MOMENT-1-small"
DEST = ROOT / "models" / "moment-1-small"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=None, help="commit sha to pin (default: resolve 'main' now and pin it)")
    args = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(REPO_ID, revision=args.revision)
    revision = info.sha
    path = snapshot_download(REPO_ID, revision=revision, local_dir=DEST,
                             # safetensors only: never ship the pickle-based pytorch_model.bin
                             allow_patterns=["*.json", "*.safetensors", "*.md", "LICENSE*"])
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(Path(path).iterdir())
             if p.is_file() and p.name not in ("PROVENANCE.json", "moment_meta.json")}
    prov = {"repo_id": REPO_ID, "revision": revision, "license": getattr(info.card_data, "license", None)
            if info.card_data else None, "files": files}
    (DEST / "PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
