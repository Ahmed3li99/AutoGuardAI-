"""Derive the limits of the 'implausible spike' checks (RPM, MAF, barometric pressure) from real data.

Data used (DERIVATION set only):
  - carOBD Toyota TRAIN + CALIBRATION trips (RPM, baro; it has no MAF)
  - VED DEV ICE cars (RPM, MAF; VED has no baro)  - datasets/splits/ved_cars_split.json
NOT used: Toyota TEST trips, VED TEST cars, all VED hybrids/plug-ins/EVs, Ford, Chrysler. Those are
the held-out data on which experiments/evaluate_plausibility.py measures the check.

For every signal: the largest value and the largest 1-second change seen, plus high quantiles
(to see whether the maximum is a lone outlier). Only seconds where both neighbours are present count
for changes (no change across gaps).

Usage:  python experiments/derive_plausibility_limits.py
Output: data/processed/plausibility_limits.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
from common import PROCESSED, load_partition, toyota_split
from evaluate_other_cars import ved_frames, ved_split

SIGNALS = ["rpm", "mass_air_flow_g_per_s", "baro_pressure_kpa"]
QS = [99.9, 99.99, 99.999]


def collect(frames, sig):
    vals, steps = [], []
    for f in frames:
        if sig not in f:
            continue
        v = f[sig].to_numpy(float)
        dv = np.abs(np.diff(v))
        vals.append(v[np.isfinite(v)])
        steps.append(dv[np.isfinite(dv)])
    v = np.concatenate(vals) if vals else np.array([])
    d = np.concatenate(steps) if steps else np.array([])
    if not v.size:
        return None
    return {"seconds": int(v.size), "max": round(float(v.max()), 2), "min": round(float(v.min()), 2),
            **{f"q{q}": round(float(np.percentile(v, q)), 2) for q in QS},
            "max_step_per_s": round(float(d.max()), 2) if d.size else None,
            **{f"step_q{q}": round(float(np.percentile(d, q)), 2) for q in QS if d.size}}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    split = toyota_split()
    toyota = list(load_partition(split["train"] + split["calibration"]).values())
    ved = [f for vid in ved_split()["dev"] for f in ved_frames(vid)]
    out = {"toyota_train_calibration": {"trips": len(toyota)}, "ved_dev_ice": {"trips": len(ved)}}
    for sig in SIGNALS:
        out["toyota_train_calibration"][sig] = collect(toyota, sig)
        out["ved_dev_ice"][sig] = collect(ved, sig)
    print(json.dumps(out, indent=1))
    (PROCESSED / "plausibility_limits.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
