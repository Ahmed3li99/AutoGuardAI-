"""'Learn your car' experiment: base expected-vs-actual model vs a PERSONAL model
learned from only a few reference trips. TEST-partition trips only (never used to
train or calibrate the base model).

Part A - same car (Toyota): personal model from k = 3 / 5 test trips, judged on
         the remaining test trips.
Adaptation settings (gate, scale clip, range source) were chosen on the
CALIBRATION partition beforehand; this script is the one-off measurement on TEST.

Part B - a SIMULATED different car: every test trip is transformed into a car
         with different gearing (RPM x 1.18), a hotter thermostat (coolant +6 °C)
         and a lower alternator set-point (voltage -0.35 V). This is an artificial
         transformation of real data, NOT a recording of a second real car.

Metrics on held-out trips: false alerts per hour (unmodified trips) and STRICT
detection of injected faults (alert on the injected signal, new vs the clean copy).
Repeated over random draws of reference trips (seeded).

Usage:  python experiments/evaluate_personal.py
Output: data/processed/personal_evaluation.json
"""
from __future__ import annotations

import json

import numpy as np

from common import PROCESSED, ROOT, load_partition, toyota_split

from ai import normal_behavior as nbm
from ai.normal_behavior import NormalBehaviorModel
from ai.personal import MIN_MINUTES, MIN_MOVING_MINUTES, adapt_base_model, moving_minutes, split_reference
from ai.consistency import run_consistency
from ai.rules_baseline import run_rules
from core.analytics.injection import InjectedEvent, changes_data, inject
from core.schema import TIME_COL

SEED = 99
DRAWS = 4
KS = (3, 5)
INJ = [("coolant_overheat", "coolant_temp_c", 180), ("voltage_drop", "battery_voltage", 60),
       ("flatline", "rpm", 60), ("spike_noise", "speed_kmh", 30)]


def other_car(df):
    d = df.copy()
    d["rpm"] = d["rpm"] * 1.18
    d["coolant_temp_c"] = d["coolant_temp_c"] + 6.0
    d["battery_voltage"] = d["battery_voltage"] - 0.35
    return d


def judge(model, trips: dict, rng) -> dict:
    """Model alone, and the app's combination (limit rules OR the model), strict matching."""
    hours = sum(len(f) for f in trips.values()) / 3600
    clean = {n: model.alerts(f) for n, f in trips.items()}
    clean_rules = {n: run_rules(f) for n, f in trips.items()}
    clean_cons = {n: run_consistency(f) for n, f in trips.items()}
    det_app = {k: [0, 0] for k, _, _ in INJ}
    det = {k: [0, 0] for k, _, _ in INJ}
    det_combined = {k: [0, 0] for k, _, _ in INJ}
    for n, df in trips.items():
        t = df[TIME_COL].to_numpy()
        for kind, sig, dur in INJ:
            if len(df) < dur + 240:
                continue
            s = float(t[0] + rng.integers(120, len(df) - dur - 60))
            ev = InjectedEvent(kind, sig, s, s + dur, "test")
            def on(alerts):
                return any(a.t_end_s >= s and a.t_start_s < s + dur and sig in a.signals for a in alerts)
            dfi = inject(df, [ev])
            if not changes_data(df, dfi, ev):     # not a valid fault label: the signal did not change
                continue
            hit, pre = on(model.alerts(dfi)), on(clean[n])
            hit_r, pre_r = on(run_rules(dfi)), on(clean_rules[n])
            hit_c, pre_c = on(run_consistency(dfi)), on(clean_cons[n])
            det[kind][0] += int(hit and not pre)
            det[kind][1] += 1
            det_combined[kind][0] += int((hit and not pre) or (hit_r and not pre_r))
            det_combined[kind][1] += 1
            det_app[kind][0] += int((hit and not pre) or (hit_r and not pre_r) or (hit_c and not pre_c))
            det_app[kind][1] += 1
    return {"false_alerts_per_hour": round(sum(len(v) for v in clean.values()) / hours, 2),
            "strict_detection": {k: v for k, v in det.items()},
            "strict_detection_with_rules": {k: v for k, v in det_combined.items()},
            "strict_detection_app": {k: v for k, v in det_app.items()}, "hours": round(hours, 2)}


def personal_from(trips: dict, names: list[str], base: NormalBehaviorModel, mode: str) -> NormalBehaviorModel:
    tr, cal = split_reference(list(range(len(names))))
    train, calib = [trips[names[i]] for i in tr], [trips[names[i]] for i in cal]
    meta = {"version": f"personal-experiment-{mode}", "scope": "vehicle"}
    if mode == "refit":            # v1 design: fit new regressors from the few trips (kept for comparison)
        return nbm.fit(train, calib, meta, min_rows=300)
    return adapt_base_model(base, train, calib, meta)      # v2 design used by the app


def main() -> None:
    split = toyota_split()
    test = load_partition(split["test"])
    base = NormalBehaviorModel.load(ROOT / "models" / "normal_behavior")
    rng0 = np.random.default_rng(SEED)
    res = {"seed": SEED, "draws": DRAWS, "test_trips": len(test), "parts": {}}
    for part, trips in (("A_same_car", test), ("B_simulated_other_car", {n: other_car(f) for n, f in test.items()})):
        out = {"base_model": judge(base, trips, np.random.default_rng(SEED))}
        for k in KS:
            runs, rejected = [], []
            names_all = sorted(trips)
            for _ in range(DRAWS):
                while True:                                         # same readiness rules as the app
                    pick = list(rng0.choice(names_all, size=k, replace=False))
                    if sum(len(trips[n]) for n in pick) < MIN_MINUTES * 60:
                        continue
                    tr_idx, _ = split_reference(list(range(len(pick))))
                    if moving_minutes([trips[pick[i]] for i in tr_idx]) >= MIN_MOVING_MINUTES:
                        break
                    rejected.append(pick)
                held = {n: f for n, f in trips.items() if n not in pick}
                runs.append({"reference": pick, "minutes": round(sum(len(trips[n]) for n in pick) / 60, 1),
                             "personal": judge(personal_from(trips, pick, base, "adapt"), held,
                                               np.random.default_rng(SEED)),
                             "personal_refit_v1": judge(personal_from(trips, pick, base, "refit"), held,
                                                        np.random.default_rng(SEED)),
                             "base_same_trips": judge(base, held, np.random.default_rng(SEED))})
            out[f"personal_k{k}"] = runs
            out[f"rejected_draws_k{k}"] = rejected      # reference sets the app would refuse (too little driving)
        res["parts"][part] = out
        print(part, "base FA/h", out["base_model"]["false_alerts_per_hour"])
        for k in KS:
            for name in ("personal", "personal_refit_v1", "base_same_trips"):
                fa = [r[name]["false_alerts_per_hour"] for r in out[f"personal_k{k}"]]
                det = {kind: sum(r[name]["strict_detection"][kind][0] for r in out[f"personal_k{k}"]) /
                       max(1, sum(r[name]["strict_detection"][kind][1] for r in out[f"personal_k{k}"]))
                       for kind, _, _ in INJ}
                comb = {kind: sum(r[name]["strict_detection_with_rules"][kind][0] for r in out[f"personal_k{k}"]) /
                        max(1, sum(r[name]["strict_detection_with_rules"][kind][1] for r in out[f"personal_k{k}"]))
                        for kind, _, _ in INJ}
                print(f"  k={k} {name:17s} FA/h mean {np.mean(fa):.2f} (min {min(fa):.2f} max {max(fa):.2f})",
                      "model", {kk: round(v, 2) for kk, v in det.items()},
                      "rules+model", {kk: round(v, 2) for kk, v in comb.items()})
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "personal_evaluation.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
