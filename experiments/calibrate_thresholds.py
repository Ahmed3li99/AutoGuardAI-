"""Fit the Isolation Forest on TRAIN trips and calibrate alert thresholds for
both anomaly models on the disjoint CALIBRATION trips (carOBD, one vehicle).
The TEST trips are not touched here.

Usage:  python experiments/calibrate_thresholds.py [--skip-moment] [--device cpu|cuda|auto]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from common import PROCESSED, ROOT, SPLIT_PATH, load_partition, toyota_split

from ai import isolation_forest_baseline as ifb
from ai.thresholds import calibrate
from core.preprocessing.features import MODEL_SIGNALS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-moment", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    today = dt.date.today().isoformat()
    split = toyota_split()
    for a in split:
        for b in split:
            assert a == b or not set(split[a]) & set(split[b]), "partition overlap"
    print({k: len(v) for k, v in split.items()})
    train = load_partition(split["train"])
    calib = load_partition(split["calibration"])
    summary = {"date": today, "split_file": str(SPLIT_PATH.relative_to(ROOT)),
               "partition_trips": {k: len(v) for k, v in split.items()},
               "partition_seconds": {"train": int(sum(len(f) for f in train.values())),
                                     "calibration": int(sum(len(f) for f in calib.values()))},
               "signals": MODEL_SIGNALS}

    # ---- Isolation Forest ------------------------------------------------
    version = f"iforest-carobd-v1-{today}"
    model = ifb.fit(list(train.values()), MODEL_SIGNALS,
                    {"version": version, "trained_on": f"toyota_carobd train partition ({len(train)} trips)",
                     "calibrated_on_dataset": "toyota_carobd", "split_file": summary["split_file"]})
    cal_scores = np.concatenate([model.score(f)[0] for f in calib.values()])
    model.threshold = calibrate(cal_scores, f"toyota_carobd calibration partition ({len(calib)} trips)")
    out_dir = ROOT / "models" / "iforest"
    model.save(out_dir)
    summary["isolation_forest"] = {"version": version, "n_train_rows": model.meta["n_train_rows"],
                                   "threshold": model.threshold.to_dict()}
    print("iforest", summary["isolation_forest"])

    # ---- MOMENT ----------------------------------------------------------
    if not args.skip_moment:
        from ai.moment_inference import ADAPTED_MODEL_DIR, BASE_MODEL_DIR, MomentScorer
        allx = pd.concat([f[MODEL_SIGNALS] for f in train.values()])
        mean, std = allx.mean().to_numpy(), allx.std().to_numpy().copy()
        std[std < 1e-6] = 1.0
        prov = json.loads((BASE_MODEL_DIR / "PROVENANCE.json").read_text(encoding="utf-8"))
        variants = [("moment", BASE_MODEL_DIR, "unmasked", "zero-shot")]
        if (ADAPTED_MODEL_DIR / "config.json").exists():
            variants.append(("moment_adapted", ADAPTED_MODEL_DIR, "masked", "carobd-adapted"))
        for key, mdir, scoring, tag in variants:
            meta = {"version": f"moment-1-small@{prov['revision'][:10]}+{tag}+thr-carobd-v1-{today}",
                    "hf_revision": prov["revision"], "weights": tag, "scoring": scoring, "signals": MODEL_SIGNALS,
                    "mean": mean.tolist(), "std": std.tolist(), "scaler_fitted_on": "toyota_carobd train partition",
                    "calibrated_on": "toyota_carobd calibration partition", "calibrated_on_dataset": "toyota_carobd",
                    "seq_len": 512, "stride": 256}
            (mdir / "moment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            scorer = MomentScorer.load(mdir, args.device)
            parts, secs = [], 0.0
            for f in calib.values():
                s, _, info = scorer.score(f)
                parts.append(s)
                secs += info["seconds"]
            thr = calibrate(np.concatenate(parts), f"toyota_carobd calibration partition ({len(calib)} trips)")
            meta["threshold"] = thr.to_dict()
            (mdir / "moment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            summary[key] = {"version": meta["version"], "device": scorer.device, "threshold": thr.to_dict(),
                            "calibration_scoring_seconds": round(secs, 2)}
            print(key, summary[key])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    p = PROCESSED / "calibration_summary.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    p.write_text(json.dumps({**old, **summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
