"""Fit the expected-vs-actual model on carOBD TRAIN trips and calibrate its
per-signal thresholds on CALIBRATION trips. TEST trips are not touched.

Usage:  python experiments/train_normal_behavior.py [--variant full|drivetrain]

The 'drivetrain' variant uses only RPM and speed (inputs and targets). It is the
model the app picks for logs that lack the other sensors (e.g. the VED fleet),
so its scores are not degraded by missing inputs.
"""
from __future__ import annotations

import datetime as dt
import json

from common import PROCESSED, ROOT, SPLIT_PATH, load_partition, toyota_split

from ai import normal_behavior as nbm


VARIANTS = {"full": (None, None, "normal_behavior", "expected-vs-actual"),
            "drivetrain": (["rpm", "speed_kmh"], ["rpm", "speed_kmh"], "normal_behavior_drivetrain",
                           "expected-vs-actual-drivetrain")}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="full")
    variant = ap.parse_args().variant
    context, targets, out_dir, name = VARIANTS[variant]
    today = dt.date.today().isoformat()
    split = toyota_split()
    train = load_partition(split["train"])
    calib = load_partition(split["calibration"])
    model = nbm.fit(list(train.values()), list(calib.values()), {
        "version": f"{name}-carobd-v1-{today}", "variant": variant,
        "trained_on": f"toyota_carobd train partition ({len(train)} trips)",
        "calibration_source": f"toyota_carobd calibration partition ({len(calib)} trips)",
        "calibrated_on_dataset": "toyota_carobd", "split_file": str(SPLIT_PATH.relative_to(ROOT))},
        context=context, targets=targets)
    model.save(ROOT / "models" / out_dir)
    summary = {t: {k: v for k, v in d.items()} for t, d in model.meta["per_target"].items()}
    print(json.dumps(summary, indent=1))
    p = PROCESSED / "calibration_summary.json"
    s = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    s["expected_vs_actual" if variant == "full" else "expected_vs_actual_drivetrain"] = {"version": model.version, "per_target": summary,
                               "threshold_quantile": nbm.THRESHOLD_Q}
    p.write_text(json.dumps(s, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
