"""Derive the 'frozen reading' (stuck sensor) limits from real driving data.

A frozen RPM reading: RPM repeats the exact same value second after second while the car is moving
(> 20 km/h) and the speed clearly changes (the engine speed of a real car must follow).
A frozen speed reading: speed repeats exactly while moving and RPM clearly changes.

Measured on the DERIVATION data only: carOBD Toyota train + calibration trips, VED DEV petrol cars.
Not used: Toyota test trips, VED test cars, hybrids, Ford, Chrysler.
Rule fixed before measuring held-out data: limit = ceil(1.25 x the longest such run seen).

Usage:  python experiments/derive_stuck_limits.py
Output: data/processed/stuck_limits.json
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
from common import PROCESSED, load_partition, toyota_split
from evaluate_other_cars import ved_frames, ved_split

MOVING = 20.0
OTHER_CHANGE = {"rpm": ("speed_kmh", 5.0), "speed_kmh": ("rpm", 200.0)}


def runs(f, sig: str) -> list[int]:
    """Lengths (s) of exact-repeat runs of `sig` while moving and while the other signal changed."""
    other, need = OTHER_CHANGE[sig]
    v = f[sig].to_numpy(float)
    o = f[other].to_numpy(float)
    spd = f["speed_kmh"].to_numpy(float)
    out, start = [], 0
    for i in range(1, len(v) + 1):
        if i < len(v) and np.isfinite(v[i]) and v[i] == v[i - 1]:
            continue
        a, b = start, i - 1
        if b - a + 1 >= 2 and np.isfinite(v[a]):
            seg_o, seg_s = o[a:b + 1], spd[a:b + 1]
            if (np.nanmin(seg_s) > MOVING and np.isfinite(seg_o).any()
                    and np.nanmax(seg_o) - np.nanmin(seg_o) >= need):
                out.append(b - a + 1)
        start = i
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    s = toyota_split()
    sources = {"toyota_train_calibration": list(load_partition(s["train"] + s["calibration"]).values()),
               "ved_dev_ice": [f for vid in ved_split()["dev"] for f in ved_frames(vid)]}
    out = {}
    for sig in ("rpm", "speed_kmh"):
        allr = []
        for name, frames in sources.items():
            r = [x for f in frames if sig in f and OTHER_CHANGE[sig][0] in f for x in runs(f, sig)]
            out.setdefault(sig, {})[name] = {"runs": len(r), "max_s": max(r) if r else 0,
                                             "q99.99_s": float(np.percentile(r, 99.99)) if r else 0}
            allr += r
        mx = max(allr) if allr else 0
        out[sig]["limit_s"] = math.ceil(1.25 * mx)
        print(sig, out[sig], flush=True)
    (PROCESSED / "stuck_limits.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
