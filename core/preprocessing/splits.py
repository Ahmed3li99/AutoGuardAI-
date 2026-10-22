"""Deterministic trip-level split for the carOBD dataset.

Whole trips (files) are assigned to train / calibration / test so no window
crosses partitions. Assignment is stratified by operating-mode category
(idle/drive/live/long/ufpe) with a fixed seed. All trips come from ONE
vehicle, so results never support claims about other vehicles."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

SPLIT_SEED = 20261022
FRACTIONS = {"train": 0.6, "calibration": 0.2, "test": 0.2}


def category_of(name: str) -> str:
    return re.sub(r"\d+$", "", Path(name).stem)


def make_trip_split(file_names: list[str], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    by_cat: dict[str, list[str]] = {}
    for n in sorted(file_names):
        by_cat.setdefault(category_of(n), []).append(n)
    out = {k: [] for k in FRACTIONS}
    rng = random.Random(seed)
    for cat in sorted(by_cat):
        trips = by_cat[cat][:]
        rng.shuffle(trips)
        n = len(trips)
        n_cal = max(1, round(n * FRACTIONS["calibration"]))
        n_test = max(1, round(n * FRACTIONS["test"]))
        out["test"] += trips[:n_test]
        out["calibration"] += trips[n_test:n_test + n_cal]
        out["train"] += trips[n_test + n_cal:]
    return {k: sorted(v) for k, v in out.items()}


def save_split(split: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": SPLIT_SEED, "fractions": FRACTIONS, **split}, indent=2),
                    encoding="utf-8")


def load_split(path: Path) -> dict[str, list[str]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: d[k] for k in FRACTIONS}
