"""Build the clean multi-car training set for the drivetrain (RPM + speed) model, with leakage checks.

Sources (never the held-out test data):
  - carOBD Toyota TRAIN / CALIBRATION partitions (existing trip split)
  - VED DEV petrol cars (datasets/splits/ved_cars_split.json), split again BY CAR into
    train cars (80 %) and calibration cars (20 %). The 132 VED TEST cars are not read for training.

Cleaning (decided from data quality only, before any model result was seen):
  - trip >= 300 s; RPM and speed present in >= 90 % of its seconds
  - values physically possible: 0 <= RPM <= 8,000 and 0 <= speed <= 250 km/h
  - engine running (RPM > 0) in >= 80 % of the seconds: no engine-off / parking logs
  - no glitch found by the consistency checks (impossible jumps/spikes, moving with 0 RPM)
  - exact duplicates removed (fingerprint of the rounded RPM + speed series)
  - at most MAX_H_PER_CAR hours per car, so a few heavy drivers do not dominate

Leakage checks (the script stops if one fails):
  - no car in more than one of train / calibration / test
  - no trip fingerprint shared between training data (Toyota + VED) and any test trip

Usage:  python experiments/build_clean_dataset.py
Output: datasets/splits/multicar_split.json (cars, trip ids, counts, reasons)
        data/processed/multicar_{train,calibration}.pkl (local 1 Hz frames, not distributed)
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from common import PROCESSED, ROOT, load_partition, toyota_split
from evaluate_other_cars import ved_frames, ved_split

from ai.consistency import run_consistency

SEED = 20261102
MIN_S = 300
COVER = 0.90
RUNNING = 0.80
MAX_H_PER_CAR = 3.0
CAL_SHARE = 0.20
OUT_SPLIT = ROOT / "datasets" / "splits" / "multicar_split.json"
GLITCH = ("_jump", "_spike", "moving_rpm0")


def fingerprint(f: pd.DataFrame) -> str:
    a = np.round(f[["rpm", "speed_kmh"]].to_numpy(float), 0)
    return hashlib.sha1(np.nan_to_num(a, nan=-1).tobytes()).hexdigest()


def check(f: pd.DataFrame) -> str | None:
    """None if the trip is clean, else the reason it is dropped."""
    if len(f) < MIN_S:
        return "shorter_than_300s"
    if f["rpm"].notna().mean() < COVER or f["speed_kmh"].notna().mean() < COVER:
        return "rpm_or_speed_missing"
    r, s = f["rpm"].dropna(), f["speed_kmh"].dropna()
    if r.min() < 0 or r.max() > 8000 or s.min() < 0 or s.max() > 250:
        return "impossible_value"
    if (r > 0).mean() < RUNNING:
        return "engine_off_most_of_trip"
    if any(a.detector.endswith(GLITCH) for a in run_consistency(f)):
        return "glitch_found"
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    split = ved_split()
    rng = np.random.default_rng(SEED)
    dev = [int(x) for x in rng.permutation(split["dev"])]
    n_cal = int(round(len(dev) * CAL_SHARE))
    cars = {"calibration": sorted(dev[:n_cal]), "train": sorted(dev[n_cal:]), "test": sorted(split["test"])}
    # leakage check 1: cars
    for a, b in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        assert not set(cars[a]) & set(cars[b]), f"car overlap {a}/{b}"

    reasons, kept, frames = Counter(), {"train": [], "calibration": []}, {"train": [], "calibration": []}
    hashes = {"train": set(), "calibration": set()}
    for part in ("train", "calibration"):
        for vid in cars[part]:
            budget = MAX_H_PER_CAR * 3600
            for i, f in enumerate(ved_frames(vid)):
                why = check(f)
                if why:
                    reasons[why] += 1
                    continue
                h = fingerprint(f)
                if h in hashes["train"] or h in hashes["calibration"]:
                    reasons["duplicate"] += 1
                    continue
                if budget <= 0:
                    reasons["car_hour_cap"] += 1
                    continue
                budget -= len(f)
                hashes[part].add(h)
                kept[part].append([vid, i])
                frames[part].append(f[["t_s", "rpm", "speed_kmh"]].reset_index(drop=True))
        print(part, len(frames[part]), "trips", round(sum(len(f) for f in frames[part]) / 3600, 1), "h", flush=True)

    # leakage check 2: no training trip equals a test trip (VED test cars, Toyota test)
    test_hashes = {fingerprint(f) for vid in cars["test"] for f in ved_frames(vid)}
    tsplit = toyota_split()
    toy = {p: load_partition(tsplit[p]) for p in ("train", "calibration", "test")}
    test_hashes |= {fingerprint(f) for f in toy["test"].values()}
    train_hashes = hashes["train"] | hashes["calibration"] | {fingerprint(f) for p in ("train", "calibration")
                                                              for f in toy[p].values()}
    overlap = train_hashes & test_hashes
    assert not overlap, f"{len(overlap)} training trips also appear in the test data"

    pd.to_pickle(frames["train"], PROCESSED / "multicar_train.pkl")        # local cache, written by this script
    pd.to_pickle(frames["calibration"], PROCESSED / "multicar_calibration.pkl")
    out = {"seed": SEED, "rules": {"min_s": MIN_S, "coverage": COVER, "engine_running": RUNNING,
                                   "max_hours_per_car": MAX_H_PER_CAR, "calibration_car_share": CAL_SHARE,
                                   "glitch_detectors": list(GLITCH)},
           "cars": cars, "kept_trips": kept, "dropped": dict(reasons),
           "hours": {p: round(sum(len(f) for f in frames[p]) / 3600, 1) for p in frames},
           "leakage_checks": {"car_overlap": 0, "trip_fingerprint_overlap_with_test": 0,
                              "test_fingerprints_checked": len(test_hashes)}}
    OUT_SPLIT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("dropped", "hours", "leakage_checks")}, indent=1))


if __name__ == "__main__":
    main()
