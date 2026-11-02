"""Held-out evaluation on the carOBD TEST partition (never used for fitting or
threshold calibration).

Part A - unmodified test trips (assumed normal operation; NO labels exist):
         alert events per hour and fraction of seconds above threshold.
Part B - SYNTHETIC injection stress test: documented anomalies are injected
         into copies of the test trips. Detection rates here measure software
         sensitivity to these specific artificial patterns only - they are NOT
         real-world fault-detection performance.
Part C - latency / memory.

Usage:  python experiments/evaluate_anomalies.py [--device cuda|cpu|auto]
Output: data/processed/evaluation_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import tracemalloc

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import PROCESSED, ROOT, load_partition, toyota_split

from ai.isolation_forest_baseline import IFModel
from ai.normal_behavior import NormalBehaviorModel
from ai.moment_inference import ADAPTED_MODEL_DIR, BASE_MODEL_DIR, MomentScorer
from ai.consistency import run_consistency
from ai.rules_baseline import run_rules
from ai.thresholds import scores_to_alerts
from core.analytics.injection import InjectedEvent, changes_data, inject, label_vector
from core.schema import TIME_COL

SEED = 7
INJECTIONS = [  # kind, signal, duration_s
    ("coolant_overheat", "coolant_temp_c", 180),
    ("voltage_drop", "battery_voltage", 60),
    ("flatline", "rpm", 60),
    ("spike_noise", "speed_kmh", 30),
    # sensor faults (targets of the sensor-consistency check)
    ("step_jump", "coolant_temp_c", 60),       # +25 C offset: an impossible 1-second jump in and out
    ("dropout", "intake_air_temp_c", 60),      # one sensor stops reporting
]
MAGNITUDE = {"step_jump": 25.0}
APP_DETECTORS = ("rules", "consistency", "expected_vs_actual")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def detect_all(df, ifm, moments: dict, nbm=None, origin="recorded"):
    """Returns {detector: (per-second scores or None, alerts, seconds)}."""
    out = {}
    t0 = time.perf_counter()
    out["rules"] = (None, run_rules(df, origin=origin), time.perf_counter() - t0)
    t0 = time.perf_counter()
    out["consistency"] = (None, run_consistency(df, origin=origin), time.perf_counter() - t0)
    t0 = time.perf_counter()
    s, c = ifm.score(df)
    out["isolation_forest"] = (s, scores_to_alerts(df, s, ifm.threshold, "if", ifm.version, c),
                               time.perf_counter() - t0)
    for key, mom in moments.items():
        t0 = time.perf_counter()
        s, c, _ = mom.score(df)
        out[key] = (s, scores_to_alerts(df, s, mom.threshold, key, mom.version, c), time.perf_counter() - t0)
    if nbm is not None:
        t0 = time.perf_counter()
        pred = nbm.predict(df)
        s, _ = nbm.score(df, pred)
        out["expected_vs_actual"] = (s, nbm.alerts(df, origin, pred=pred), time.perf_counter() - t0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    split = toyota_split()
    test = load_partition(split["test"])
    ifm = IFModel.load(ROOT / "models" / "iforest")
    moments = {}
    for key, mdir in (("moment", BASE_MODEL_DIR), ("moment_adapted", ADAPTED_MODEL_DIR)):
        try:
            moments[key] = MomentScorer.load(mdir, args.device)
        except Exception as e:
            print(f"{key} unavailable:", e)
    try:
        nbm = NormalBehaviorModel.load(ROOT / "models" / "normal_behavior")
    except FileNotFoundError:
        nbm = None
    dets = ["rules", "consistency", "isolation_forest", *moments] + (["expected_vs_actual"] if nbm else [])
    scorers = {"isolation_forest": ifm, **moments, "expected_vs_actual": nbm}
    res = {"test_trips": split["test"], "n_test_trips": len(test),
           "test_seconds": int(sum(len(f) for f in test.values())),
           "versions": {d: getattr(scorers.get(d), "version", {"rules": "rules-v1", "consistency": "consistency-v1"}.get(d)) for d in dets},
           "moment_device": next(iter(moments.values())).device if moments else None}

    # ---- A: unmodified test trips --------------------------------------
    a = {d: {"alerts": 0, "alert_seconds": 0, "above_thr_seconds": 0, "scored_seconds": 0,
             "by_category": {}} for d in dets}
    timing = {d: [] for d in dets}
    clean_alerts: dict[str, dict] = {}       # trip -> detector -> alerts on the UNMODIFIED trip
    for name, df in test.items():
        cat = name.rstrip(".csv").rstrip("0123456789")
        out = detect_all(df, ifm, moments, nbm)
        clean_alerts[name] = {d: out[d][1] for d in dets}
        for d in dets:
            sc, alerts, sec = out[d]
            timing[d].append((len(df), sec))
            a[d]["alerts"] += len(alerts)
            a[d]["alert_seconds"] += int(sum(x.t_end_s - x.t_start_s + 1 for x in alerts))
            bc = a[d]["by_category"].setdefault(cat, {"alerts": 0, "seconds": 0})
            bc["alerts"] += len(alerts)
            bc["seconds"] += len(df)
            if sc is not None:
                thr = scorers[d].threshold.attention
                ok = np.isfinite(sc)
                a[d]["scored_seconds"] += int(ok.sum())
                a[d]["above_thr_seconds"] += int((sc[ok] > thr).sum())
    hours = res["test_seconds"] / 3600
    for d in dets:
        a[d]["alerts_per_hour"] = round(a[d]["alerts"] / hours, 2)
        a[d]["fraction_seconds_in_alert"] = round(a[d]["alert_seconds"] / res["test_seconds"], 4)
        if a[d]["scored_seconds"]:
            a[d]["fraction_seconds_above_attention_thr"] = round(a[d]["above_thr_seconds"] / a[d]["scored_seconds"], 4)
        for bc in a[d]["by_category"].values():
            bc["alerts_per_hour"] = round(bc["alerts"] / (bc["seconds"] / 3600), 2)
    res["A_unmodified_test_trips"] = a

    # ---- B: synthetic injection stress test ----------------------------
    rng = np.random.default_rng(SEED)
    # lenient = any alert overlapping the injected interval (original metric, kept for comparison)
    # strict  = overlapping alert that (1) names the injected signal and (2) is NEW, i.e. no alert
    #           of the same detector on that signal overlapped this interval in the unmodified trip
    b = {k: {d: {"injected": 0, "detected": 0, "detected_strict": 0, "delays_s": [], "delays_strict_s": [],
                 "auroc": [], "ap": []} for d in dets}
         for k, _, _ in INJECTIONS}
    invalid = {k: 0 for k, _, _ in INJECTIONS}
    app = {k: [0, 0] for k, _, _ in INJECTIONS}     # what the app's verdict uses: rules OR consistency OR model
    for name, df in test.items():
        t = df[TIME_COL].to_numpy()
        for kind, sig, dur in INJECTIONS:
            if len(df) < dur + 240:
                continue
            start = float(t[0] + rng.integers(120, len(df) - dur - 60))
            ev = InjectedEvent(kind, sig, start, start + dur, f"stress test {kind}", MAGNITUDE.get(kind))
            dfi = inject(df, [ev])
            if not changes_data(df, dfi, ev):          # not a valid fault label (see changes_data)
                invalid[kind] += 1
                continue
            y = label_vector(dfi[TIME_COL].to_numpy(), [ev])
            out = detect_all(dfi, ifm, moments, nbm, origin="synthetic")
            app_hit = False
            for d in dets:
                sc, alerts, _ = out[d]
                r = b[kind][d]
                r["injected"] += 1
                def overlaps(x):
                    return x.t_end_s >= ev.start_s and x.t_start_s < ev.end_s

                hits = [x for x in alerts if overlaps(x)]
                if hits:
                    r["detected"] += 1
                    r["delays_s"].append(max(0.0, min(x.t_start_s for x in hits) - ev.start_s))
                pre_existing = any(overlaps(x) and sig in x.signals for x in clean_alerts[name][d])
                strict = [x for x in hits if sig in x.signals] if not pre_existing else []
                if strict:
                    r["detected_strict"] += 1
                    r["delays_strict_s"].append(max(0.0, min(x.t_start_s for x in strict) - ev.start_s))
                    app_hit |= d in APP_DETECTORS
                if sc is not None:
                    ok = np.isfinite(sc)
                    if y[ok].any() and (~y[ok]).any():
                        r["auroc"].append(roc_auc_score(y[ok], sc[ok]))
                        r["ap"].append(average_precision_score(y[ok], sc[ok]))
            app[kind][0] += int(app_hit)
            app[kind][1] += 1
    for kind in b:
        for d, r in b[kind].items():
            r["event_detection_rate"] = round(r["detected"] / r["injected"], 3) if r["injected"] else None
            r["wilson95"] = wilson(r["detected"], r["injected"])
            r["median_delay_s"] = float(np.median(r["delays_s"])) if r["delays_s"] else None
            r["strict_detection_rate"] = round(r["detected_strict"] / r["injected"], 3) if r["injected"] else None
            r["strict_wilson95"] = wilson(r["detected_strict"], r["injected"])
            r["strict_median_delay_s"] = float(np.median(r["delays_strict_s"])) if r["delays_strict_s"] else None
            r["mean_auroc"] = round(float(np.mean(r["auroc"])), 3) if r["auroc"] else None
            r["mean_average_precision"] = round(float(np.mean(r["ap"])), 3) if r["ap"] else None
            for k in ("delays_s", "delays_strict_s", "auroc", "ap"):
                r.pop(k)
    res["B_synthetic_injection_stress_test"] = b
    res["B_invalid_labels_skipped"] = dict(invalid)
    res["B_app_combined_strict"] = {k: {"detected": h, "injected": n, "rate": round(h / n, 3) if n else None,
                                        "wilson95": wilson(h, n)} for k, (h, n) in app.items()}

    # ---- C: latency / memory -------------------------------------------
    c = {}
    for d in dets:
        n = sum(x for x, _ in timing[d])
        s = sum(y for _, y in timing[d])
        c[d] = {"seconds_total": round(s, 3), "ms_per_hour_of_data": round(1000 * s / (n / 3600), 1)}
    big = max(test.values(), key=len)
    for key, mom in moments.items():
        import torch
        if mom.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mom.score(big)
            c[key]["cuda_peak_mb_longest_trip"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        cpu = MomentScorer.load(BASE_MODEL_DIR if key == "moment" else ADAPTED_MODEL_DIR, "cpu")
        tracemalloc.start()
        t0 = time.perf_counter()
        cpu.score(big)
        c[key]["cpu_seconds_longest_trip"] = round(time.perf_counter() - t0, 2)
        c[key]["cpu_python_heap_peak_mb"] = round(tracemalloc.get_traced_memory()[1] / 2**20, 1)
        tracemalloc.stop()
        c[key]["longest_trip_seconds"] = len(big)
    if nbm is not None:
        t0 = time.perf_counter()
        nbm.score(big)
        c["expected_vs_actual"]["cpu_seconds_longest_trip"] = round(time.perf_counter() - t0, 2)
    c["cpu_count"] = os.cpu_count()
    res["C_latency"] = c
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "evaluation_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "test_trips"}, indent=1))


if __name__ == "__main__":
    main()
