"""Train the drivetrain (RPM + speed) expected-vs-actual model on MANY cars instead of one Toyota.

Train:        carOBD Toyota train partition + clean VED train cars   (experiments/build_clean_dataset.py)
Calibration:  carOBD Toyota calibration partition + clean VED calibration cars (other cars)
Test data (Toyota test trips, the 132 VED test cars, Ford, Chrysler) is not read here.

Overfitting check: median |error| on the training cars vs on the calibration cars (cars the
regressors never saw). A large gap would mean the model memorised the training cars.

Usage:  python experiments/train_multicar_drivetrain.py
Output: models/normal_behavior_drivetrain/  (the previous Toyota-only model is kept in
        models/normal_behavior_drivetrain_toyota_only/ for comparison)
        data/processed/multicar_training.json
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import sys

import numpy as np
import pandas as pd
from common import PROCESSED, ROOT, load_partition, toyota_split

from ai import normal_behavior as nbm

OUT = ROOT / "models" / "normal_behavior_drivetrain"
OLD = ROOT / "models" / "normal_behavior_drivetrain_toyota_only"


def median_abs_error(model, frames, target):
    errs = []
    for f in frames:
        p = model.predict(f)
        if f"{target}__expected" in p:
            e = (f[target] - p[f"{target}__expected"]).abs().to_numpy()
            errs.append(e[np.isfinite(e)])
    e = np.concatenate(errs)
    return round(float(np.median(e)), 2)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    s = toyota_split()
    cols = ["t_s", "rpm", "speed_kmh"]
    toy_tr = [f[cols] for f in load_partition(s["train"]).values()]
    toy_cal = [f[cols] for f in load_partition(s["calibration"]).values()]
    ved_tr = pd.read_pickle(PROCESSED / "multicar_train.pkl")          # written by build_clean_dataset.py
    ved_cal = pd.read_pickle(PROCESSED / "multicar_calibration.pkl")
    train, calib = toy_tr + ved_tr, toy_cal + ved_cal
    today = dt.date.today().isoformat()
    split = json.loads((ROOT / "datasets" / "splits" / "multicar_split.json").read_text(encoding="utf-8"))
    model = nbm.fit(train, calib, {
        "version": f"expected-vs-actual-drivetrain-multicar-v2-{today}", "variant": "drivetrain",
        "trained_on": f"toyota_carobd train ({len(toy_tr)} trips) + VED {len(split['cars']['train'])} cars "
                      f"({len(ved_tr)} clean trips)",
        "calibration_source": f"toyota_carobd calibration ({len(toy_cal)} trips) + VED "
                              f"{len(split['cars']['calibration'])} other cars ({len(ved_cal)} trips)",
        "calibrated_on_dataset": "multi_car", "split_file": "datasets/splits/multicar_split.json"},
        context=["rpm", "speed_kmh"], targets=["rpm", "speed_kmh"])
    if OUT.exists() and not OLD.exists():
        shutil.copytree(OUT, OLD)                    # keep the Toyota-only model for the comparison
    model.save(OUT)
    report = {"version": model.version, "hours": {"train": round(sum(len(f) for f in train) / 3600, 1),
                                                  "calibration": round(sum(len(f) for f in calib) / 3600, 1)},
              "per_target": model.meta["per_target"], "overfitting_check": {}}
    for t in ("rpm", "speed_kmh"):
        report["overfitting_check"][t] = {
            "median_abs_error_train_cars": median_abs_error(model, ved_tr[::5], t),
            "median_abs_error_unseen_calibration_cars": median_abs_error(model, ved_cal, t),
            "median_abs_error_toyota_train": median_abs_error(model, toy_tr, t),
            "median_abs_error_toyota_calibration": median_abs_error(model, toy_cal, t)}
    (PROCESSED / "multicar_training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
