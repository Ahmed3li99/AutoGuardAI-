"""Choose how far 'learn your car' may widen a signal's noise margin (ai/personal.py: NOISE_UP_TO).

Why: on real cars other than the Toyota, the gap between actual and expected RPM/speed is 2-6x wider
than on the Toyota (other gearing, automatic gearboxes), so the base margins raise ~20 alerts/h.
The first design only allowed 0.8-1.2x (chosen on a SIMULATED other car, which was too mild).

Data used: VED DEV cars only (half of the ICE cars, datasets/splits/ved_cars_split.json) and the
Toyota CALIBRATION partition. VED TEST cars, Ford, Chrysler and Toyota TEST trips are not touched.

Rule, fixed before running: take the SMALLEST candidate whose pooled false-alert rate on the dev
cars' later trips is <= 2 per hour (smaller = keeps more sensitivity); if none reaches it, the one
with the fewest false alerts. Toyota calibration (same car) is reported as a safety check.

Usage:  python experiments/select_cross_car_settings.py
Output: data/processed/cross_car_selection.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
from common import PROCESSED, load_partition, toyota_split
from evaluate_other_cars import adapt, judge, ved_car_sets, ved_split
from evaluate_personal import judge as judge_toyota

from ai.model_registry import NBM_DRIVETRAIN_DIR
from ai.normal_behavior import NormalBehaviorModel

CANDIDATES = [0.0, 1.5, 2.0, 3.0, 5.0]       # 0 = the first design (adapted signals only, 0.8-1.2x)
TARGET_FA_PER_H = 2.0
DEV_EVAL_TRIPS = 20


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    base = NormalBehaviorModel.load(NBM_DRIVETRAIN_DIR)
    cars = list(ved_car_sets(ved_split()["dev"], skipped := Counter(), DEV_EVAL_TRIPS))
    print(f"dev cars ready: {len(cars)} (skipped {dict(skipped)})", flush=True)
    res = {"candidates": [], "dev_cars": len(cars), "skipped": dict(skipped), "rule": __doc__.split("Rule")[1][:260]}
    rows = {"base": [judge(base, ev, np.random.default_rng(vid)) for vid, _, ev in cars]}
    for c in CANDIDATES:
        rows[c] = [judge(adapt(base, ref, f"dev{vid}", noise_up_to=c), ev, np.random.default_rng(vid))
                   for vid, ref, ev in cars]
    # Toyota calibration partition, full model: the same car must not get worse
    from ai.model_registry import NBM_DIR
    from ai.personal import adapt_base_model, split_reference
    full = NormalBehaviorModel.load(NBM_DIR)
    cal = load_partition(toyota_split()["calibration"])
    names = sorted(cal)
    rng = np.random.default_rng(5)
    draws = [list(rng.choice(names, 4, replace=False)) for _ in range(3)]
    for name, r in rows.items():
        h = sum(x["hours"] for x in r)
        det = {k: [sum(x["strict_detection_with_consistency"][k][0] for x in r),
                   sum(x["strict_detection_with_consistency"][k][1] for x in r)] for k in r[0]["strict_detection"]}
        entry = {"noise_up_to": name, "dev_hours": round(h, 1),
                 "pooled_false_alerts_per_hour": round(sum(x["alerts"] for x in r) / h, 2),
                 "median_car_false_alerts_per_hour": round(float(np.median([x["false_alerts_per_hour"] for x in r])), 2),
                 "strict_detection_with_consistency": {k: round(a / n, 3) for k, (a, n) in det.items()}}
        if name != "base":
            fa = []
            for d in draws:
                tr, cl = split_reference(list(range(len(d))))
                m = adapt_base_model(full, [cal[d[i]] for i in tr], [cal[d[i]] for i in cl],
                                     {"version": "sel", "scope": "vehicle"}, noise_up_to=name)
                held = {n: f for n, f in cal.items() if n not in d}
                fa.append(judge_toyota(m, held, np.random.default_rng(5)))
            entry["toyota_calibration_same_car"] = {
                "false_alerts_per_hour": round(float(np.mean([x["false_alerts_per_hour"] for x in fa])), 2),
                "strict_detection_with_rules": {k: round(sum(x["strict_detection_with_rules"][k][0] for x in fa) /
                                                         max(1, sum(x["strict_detection_with_rules"][k][1] for x in fa)), 3)
                                                for k in fa[0]["strict_detection_with_rules"]}}
        res["candidates"].append(entry)
        print(json.dumps(entry), flush=True)
    ok = [e for e in res["candidates"] if e["noise_up_to"] not in ("base", 0.0)
          and e["pooled_false_alerts_per_hour"] <= TARGET_FA_PER_H]
    pool = ok or [e for e in res["candidates"] if e["noise_up_to"] != "base"]
    pick = (min(ok, key=lambda e: e["noise_up_to"]) if ok
            else min(pool, key=lambda e: e["pooled_false_alerts_per_hour"]))
    res["chosen"] = pick["noise_up_to"]
    print("chosen NOISE_UP_TO =", res["chosen"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "cross_car_selection.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
