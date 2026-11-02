"""Cross-car validation: the models were trained on ONE Toyota (carOBD). Here they are run,
unchanged, on REAL recordings of other cars, and 'learn your car' is tested on them.

Part V - VED (Vehicle Energy Dataset, Ann Arbor, 2017-18): hundreds of cars, but only RPM and
         speed (plus MAF/OAT/trims) -> the app picks the DRIVETRAIN model (RPM and speed).
         Per ICE car: reference = its EARLIEST trips, taken until the app's readiness rules are
         met (>= 3 trips, >= 30 min, >= 10 min above 20 km/h in the learning part, at most 10 trips);
         evaluation = its LATER trips. Base model vs the car's personal model:
           - alerts per hour on the later trips (no faults known -> treated as false alerts)
           - RPM prediction error while moving
           - STRICT detection of injected faults (frozen RPM, RPM offset, noisy speed)
         The sensor-consistency check is run on every trip of every car type (ICE / HEV / PHEV / EV)
         to see what it says about real data, especially 'moving with RPM = 0' on hybrids.
Part F - APMonitor Ford (2014 Focus per the dataset page): base models on the long highway file
         (auto_trip), its real glitch values vs the consistency check, and a personal drivetrain
         model learned from the separate 2.6 h file (automotive.txt, split into 3 parts).
Part C - APMonitor Chrysler Pacifica: one 67-min trip, base models only (one trip is not enough
         for 'learn your car' under the app's own rules).

Nothing here was used to train, calibrate or select any setting.

Usage:  python experiments/evaluate_other_cars.py [--max-cars N]
Output: data/processed/other_cars_evaluation.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from common import PROCESSED, ROOT

from ai.consistency import run_consistency
from ai.model_registry import NBM_DIR, NBM_DRIVETRAIN_DIR, choose_variant

DRIVETRAIN_DIR = NBM_DRIVETRAIN_DIR
from ai.normal_behavior import NormalBehaviorModel
from ai.personal import (MIN_MINUTES, MIN_MOVING_MINUTES, MIN_TRIPS, adapt_base_model, moving_minutes,
                         split_reference)
from ai.rules_baseline import run_rules
from core.analytics.injection import InjectedEvent, changes_data, inject
from core.data_sources.files import RecordedFileSource
from core.preprocessing.resample import resample_1hz
from core.schema import TIME_COL

SEED = 7
VED_DIR = ROOT / "data" / "raw" / "ved" / "by_vehicle"
APM = ROOT / "data" / "raw" / "apmonitor"
MIN_TRIP_S = 120            # shorter VED trips are skipped (a parked-car blip says nothing)
MAX_REF_TRIPS = 10
MAX_EVAL_TRIPS = 40
INJ = [("flatline", "rpm", 60, None), ("step_jump", "rpm", 60, 600.0), ("spike_noise", "speed_kmh", 30, None)]


def hours(frames) -> float:
    return sum(len(f) for f in frames) / 3600


def rpm_error(model, frames) -> float | None:
    """Median |actual - expected| RPM while moving (> 20 km/h)."""
    errs = []
    for f in frames:
        p = model.predict(f)
        if "rpm__expected" not in p:
            continue
        m = (f["speed_kmh"] > 20) & f["rpm"].notna()
        errs.append((f.loc[m, "rpm"] - p.loc[m, "rpm__expected"]).abs().to_numpy())
    e = np.concatenate(errs) if errs else np.array([])
    e = e[np.isfinite(e)]
    return round(float(np.median(e)), 1) if e.size else None


def judge(model, frames: list[pd.DataFrame], rng) -> dict:
    """False alerts per hour (unmodified trips) + strict detection, model alone and model OR the
    consistency check (the two detectors the app runs on RPM/speed-only data)."""
    clean = [model.alerts(f) for f in frames]
    clean_c = [run_consistency(f) for f in frames]
    det = {k: [0, 0] for k, *_ in INJ}
    det_c = {k: [0, 0] for k, *_ in INJ}
    for f, ca, cc in zip(frames, clean, clean_c):
        t = f[TIME_COL].to_numpy()
        for kind, sig, dur, mag in INJ:
            if len(f) < dur + 240 or f[sig].notna().mean() < 0.8:
                continue
            s = float(t[0] + rng.integers(120, len(f) - dur - 60))
            ev = InjectedEvent(kind, sig, s, s + dur, "test", mag)
            dfi = inject(f, [ev])
            if not changes_data(f, dfi, ev):     # not a valid fault label: the signal did not change
                continue

            def on(alerts):
                return any(a.t_end_s >= s and a.t_start_s < s + dur and sig in a.signals for a in alerts)
            hit = on(model.alerts(dfi)) and not on(ca)
            hit_c = on(run_consistency(dfi)) and not on(cc)
            det[kind][0] += int(hit)
            det[kind][1] += 1
            det_c[kind][0] += int(hit or hit_c)
            det_c[kind][1] += 1
    h = hours(frames)
    return {"hours": round(h, 2), "alerts": sum(len(a) for a in clean),
            "false_alerts_per_hour": round(sum(len(a) for a in clean) / h, 2) if h else None,
            "rpm_error_moving": rpm_error(model, frames),
            "strict_detection": det, "strict_detection_with_consistency": det_c}


def pick_reference(frames: list[pd.DataFrame]) -> int | None:
    """Smallest number of earliest trips that passes the app's readiness rules."""
    for k in range(MIN_TRIPS, min(MAX_REF_TRIPS, len(frames) - 1) + 1):
        ref = frames[:k]
        if hours(ref) * 60 < MIN_MINUTES:
            continue
        tr, _ = split_reference(list(range(k)))
        if moving_minutes([ref[i] for i in tr]) >= MIN_MOVING_MINUTES:
            return k
    return None


def adapt(base, ref: list[pd.DataFrame], tag: str, noise_up_to: float | None = None):
    tr, cal = split_reference(list(range(len(ref))))
    return adapt_base_model(base, [ref[i] for i in tr], [ref[i] for i in cal],
                            {"version": f"personal-{tag}", "scope": "vehicle"}, noise_up_to=noise_up_to)


# ---------------------------------------------------------------- Part V: VED
VED_SPLIT = ROOT / "datasets" / "splits" / "ved_cars_split.json"


def ved_index() -> dict[int, dict]:
    return {int(k): v for k, v in json.loads((VED_DIR / "index.json").read_text(encoding="utf-8")).items()}


def ved_split() -> dict[str, list[int]]:
    """ICE cars split once (seeded) into DEV cars (used to choose settings) and TEST cars
    (measured once). Never mixed."""
    if VED_SPLIT.exists():
        return json.loads(VED_SPLIT.read_text(encoding="utf-8"))
    ice = sorted(v for v, i in ved_index().items() if i.get("type") == "ICE")
    perm = [int(x) for x in np.random.default_rng(20261023).permutation(ice)]
    split = {"seed": 20261023, "dev": sorted(perm[:len(perm) // 2]), "test": sorted(perm[len(perm) // 2:])}
    VED_SPLIT.write_text(json.dumps(split, indent=1), encoding="utf-8")
    return split


def ved_frames(vid: int) -> list[pd.DataFrame]:
    """Usable 1 Hz trips of one car, oldest first."""
    d = pd.read_pickle(VED_DIR / f"veh_{vid}.pkl")      # written locally by prepare_ved.py
    order = sorted(d["trips"], key=lambda tr: (d["first_day"][tr], tr))
    frames = [d["trips"][tr].astype("float64") for tr in order if len(d["trips"][tr]) >= MIN_TRIP_S]
    return [f for f in frames if "rpm" in f and "speed_kmh" in f
            and f["rpm"].notna().mean() > 0.5 and f["speed_kmh"].notna().mean() > 0.5]


def ved_car_sets(ids: list[int], skipped: Counter, max_eval: int = MAX_EVAL_TRIPS):
    """(vid, reference trips, later trips) for cars that pass the app's readiness rules."""
    for vid in ids:
        frames = ved_frames(vid)
        k = pick_reference(frames)
        if k is None:
            skipped["not_ready"] += 1
            continue
        ref, ev = frames[:k], frames[k:k + max_eval]
        if hours(ev) < 0.5:
            skipped["less_than_30_min_after_reference"] += 1
            continue
        yield vid, ref, ev


def ved_consistency(index: dict[int, dict]) -> dict:
    """Consistency check on every usable trip of every car type (its thresholds were never tuned on VED)."""
    cons = defaultdict(lambda: {"cars": 0, "hours": 0.0, "by_detector": Counter()})
    for vid, info in sorted(index.items()):
        frames = ved_frames(vid)
        c = cons[info.get("type", "unknown")]
        c["cars"] += 1
        c["hours"] += hours(frames)
        for f in frames:
            for a in run_consistency(f):
                c["by_detector"][a.detector] += 1
    return {t: {"cars": v["cars"], "hours": round(v["hours"], 1),
                "alerts_per_hour": round(sum(v["by_detector"].values()) / v["hours"], 3) if v["hours"] else None,
                "per_hour_by_detector": {k: round(n / v["hours"], 3) for k, n in v["by_detector"].most_common()}}
            for t, v in cons.items()}


def part_ved(max_cars: int | None) -> dict:
    index = ved_index()
    split = ved_split()
    base = NormalBehaviorModel.load(DRIVETRAIN_DIR)
    rng = np.random.default_rng(SEED)
    cars, skipped = [], Counter()
    ids = split["test"][:max_cars] if max_cars else split["test"]
    for vid, ref, ev in ved_car_sets(ids, skipped):
        info = index[vid]
        pers = adapt(base, ref, f"ved{vid}")
        seed = int(rng.integers(1 << 30))
        cars.append({"veh_id": vid, "class": info.get("class"), "engine": info.get("engine"),
                     "transmission": info.get("transmission"), "ref_trips": len(ref),
                     "ref_minutes": round(hours(ref) * 60, 1), "eval_trips": len(ev),
                     "car_map": pers.meta.get("car_map", {}),
                     "personal_scale": {t: round(m.scale, 2) for t, m in pers.models.items()},
                     "base": judge(base, ev, np.random.default_rng(seed)),
                     "personal": judge(pers, ev, np.random.default_rng(seed))})
        b, p = cars[-1]["base"], cars[-1]["personal"]
        print(f"VED {vid:4d} k={len(ref)} eval {b['hours']:5.1f} h  FA/h base {b['false_alerts_per_hour']:6.2f} "
              f"personal {p['false_alerts_per_hour']:6.2f}  rpm err {b['rpm_error_moving']} -> "
              f"{p['rpm_error_moving']}  scale {cars[-1]['personal_scale']}", flush=True)
    return {"ice_cars_in_dataset": len(split["dev"]) + len(split["test"]), "test_cars": len(split["test"]),
            "evaluated_test_cars": len(cars), "skipped_test_cars": dict(skipped),
            "summary": summarize_cars(cars), "consistency_by_type": ved_consistency(index), "cars": cars}


def summarize_cars(cars: list[dict]) -> dict:
    if not cars:
        return {}
    out = {}
    for name in ("base", "personal"):
        h = sum(c[name]["hours"] for c in cars)
        fa = [c[name]["false_alerts_per_hour"] for c in cars]
        err = [c[name]["rpm_error_moving"] for c in cars if c[name]["rpm_error_moving"] is not None]
        det = {k: [sum(c[name]["strict_detection"][k][0] for c in cars),
                   sum(c[name]["strict_detection"][k][1] for c in cars)] for k, *_ in INJ}
        det_c = {k: [sum(c[name]["strict_detection_with_consistency"][k][0] for c in cars),
                     sum(c[name]["strict_detection_with_consistency"][k][1] for c in cars)] for k, *_ in INJ}
        out[name] = {"hours": round(h, 1),
                     "pooled_false_alerts_per_hour": round(sum(c[name]["alerts"] for c in cars) / h, 2),
                     "median_car_false_alerts_per_hour": round(float(np.median(fa)), 2),
                     "p90_car_false_alerts_per_hour": round(float(np.percentile(fa, 90)), 2),
                     "cars_under_1_alert_per_hour": sum(x < 1 for x in fa),
                     "median_rpm_error_moving": round(float(np.median(err)), 1) if err else None,
                     "strict_detection": {k: {"hit": a, "n": n, "rate": round(a / n, 3) if n else None}
                                          for k, (a, n) in det.items()},
                     "strict_detection_with_consistency": {k: {"hit": a, "n": n, "rate": round(a / n, 3) if n else None}
                                                           for k, (a, n) in det_c.items()}}
    out["cars_where_personal_has_fewer_alerts"] = sum(
        c["personal"]["false_alerts_per_hour"] < c["base"]["false_alerts_per_hour"] for c in cars)
    out["cars_where_personal_has_more_alerts"] = sum(
        c["personal"]["false_alerts_per_hour"] > c["base"]["false_alerts_per_hour"] for c in cars)
    out["cars_left_unchanged_by_gate"] = sum(not c["car_map"] for c in cars)
    return out


# ---------------------------------------------------------------- APMonitor helpers
def load_1hz(path) -> pd.DataFrame:
    s = RecordedFileSource(path).read_session()
    return resample_1hz(s.data, s.signals)[0]


def segments_by_gap(df: pd.DataFrame, sig: str = "rpm", gap_s: int = 60, min_s: int = 120) -> list[pd.DataFrame]:
    """Split a long log into continuous pieces where `sig` keeps reporting."""
    ok = df[sig].notna().to_numpy()
    out, start, miss = [], None, 0
    for i, v in enumerate(ok):
        if v:
            start = i if start is None else start
            miss = 0
        elif start is not None:
            miss += 1
            if miss > gap_s:
                out.append(df.iloc[start:i - miss + 1])
                start, miss = None, 0
    if start is not None:
        out.append(df.iloc[start:len(df) - miss])
    return [p.reset_index(drop=True) for p in out if len(p) >= min_s]


GLITCH = {"rpm": 7500, "coolant_temp_c": 140, "intake_air_temp_c": 90, "mass_air_flow_g_per_s": 300,
          "baro_pressure_kpa": 110, "catalyst_temp_c": 1100}


def glitch_events(df: pd.DataFrame) -> list[tuple[float, float, str]]:
    """Physically implausible readings (values no healthy Focus can produce) - the real data errors."""
    t = df[TIME_COL].to_numpy()
    ev = []
    for sig, hi in GLITCH.items():
        if sig not in df:
            continue
        m = (df[sig] > hi).to_numpy()
        i = 0
        while i < len(m):
            if m[i]:
                j = i
                while j + 1 < len(m) and m[j + 1]:
                    j += 1
                ev.append((float(t[i]), float(t[j]), sig))
                i = j + 1
            else:
                i += 1
    return ev


def run_all(name: str, models: dict, pieces: list[pd.DataFrame]) -> dict:
    h = hours(pieces)
    out = {"hours": round(h, 2), "pieces": len(pieces),
           "signals": sorted({c for p in pieces for c in p.columns if c != TIME_COL and p[c].notna().any()})}
    out["variant_the_app_picks"] = choose_variant(out["signals"])
    for mname, m in models.items():
        n = sum(len(m.alerts(p)) for p in pieces)
        out[f"{mname}_alerts_per_hour"] = round(n / h, 2)
        out[f"{mname}_rpm_error_moving"] = rpm_error(m, pieces)
    rules = [a for p in pieces for a in run_rules(p)]
    cons = [a for p in pieces for a in run_consistency(p)]
    out["rules_alerts"] = dict(Counter(a.detector for a in rules))
    out["consistency_alerts"] = dict(Counter(a.detector for a in cons))
    print(name, json.dumps({k: v for k, v in out.items() if k != "signals"}), flush=True)
    return out


def glitch_check(pieces: list[pd.DataFrame]) -> dict:
    """Which real glitch events are flagged by the consistency check or the limit rules."""
    n = flagged_c = flagged_any = 0
    by_sig = Counter()
    for p in pieces:
        cons, rules = run_consistency(p), run_rules(p)
        for s, e, sig in glitch_events(p):
            n += 1
            by_sig[sig] += 1
            hit_c = any(a.t_start_s <= e + 1 and a.t_end_s >= s - 1 for a in cons)
            hit_r = any(a.t_start_s <= e + 1 and a.t_end_s >= s - 1 and sig in a.signals for a in rules)
            flagged_c += hit_c
            flagged_any += hit_c or hit_r
    return {"glitch_events": n, "by_signal": dict(by_sig), "flagged_by_consistency": flagged_c,
            "flagged_by_consistency_or_rules": flagged_any, "thresholds": GLITCH}


CHUNK_S = 720


def chunk_adapt(name: str, base, pieces: list[pd.DataFrame]) -> dict:
    """Within ONE recording: cut into 12-min chunks, learn from the first 3 (36 min, the app's
    minimum is 3 trips / 30 min), judge on the rest. Optimistic: same day, same road type."""
    chunks = [p.iloc[i:i + CHUNK_S].reset_index(drop=True) for p in pieces for i in range(0, len(p), CHUNK_S)]
    chunks = [c for c in chunks if len(c) >= 300]
    ref, ev = chunks[:3], chunks[3:]
    pers = adapt(base, ref, name)
    return {"reference_minutes": round(hours(ref) * 60, 1), "eval_hours": round(hours(ev), 2),
            "car_map": pers.meta.get("car_map", {}),
            "base_alerts_per_hour_on_eval": round(sum(len(base.alerts(c)) for c in ev) / hours(ev), 2),
            "personal_alerts_per_hour_on_eval": round(sum(len(pers.alerts(c)) for c in ev) / hours(ev), 2),
            "base_rpm_error_moving": rpm_error(base, ev), "personal_rpm_error_moving": rpm_error(pers, ev)}


def part_ford(full, dt_base) -> dict:
    trip = segments_by_gap(load_1hz(APM / "auto_trip" / "auto_trip.csv"))
    ref_log = load_1hz(APM / "automotive.txt")
    ref = [ref_log.iloc[i * len(ref_log) // 3:(i + 1) * len(ref_log) // 3].reset_index(drop=True) for i in range(3)]   # one 2.6 h log -> 3 "trips"
    out = {"auto_trip": run_all("ford auto_trip", {"full": full, "drivetrain": dt_base}, trip),
           "auto_trip_glitches": glitch_check(trip),
           "reference_file": {"file": "automotive.txt", "minutes": round(hours(ref) * 60, 1),
                              "split_into": 3, "moving_minutes_learning_part":
                              round(moving_minutes(ref[:2]), 1)}}
    pers = adapt(dt_base, ref, "ford")
    out["personal_drivetrain"] = {"car_map": pers.meta.get("car_map", {}),
                                  **run_all("ford auto_trip personal", {"personal": pers}, trip)}
    out["personal_full_same_recording"] = chunk_adapt("ford-full", full, trip)
    print("ford full chunks", out["personal_full_same_recording"], flush=True)
    warm = load_1hz(APM / "auto_warmup.txt")
    out["auto_warmup"] = {"seconds": len(warm), "coverage": {c: round(float(warm[c].notna().mean()), 3)
                                                             for c in warm.columns if c != TIME_COL},
                          "ambient_values": sorted(warm["ambient_air_temp_c"].dropna().unique().tolist())
                          if "ambient_air_temp_c" in warm else None}
    return out


def part_chrysler(full, dt_base) -> dict:
    trip = segments_by_gap(load_1hz(APM / "auto_iowa.txt"))
    out = {"auto_iowa": run_all("chrysler auto_iowa", {"full": full, "drivetrain": dt_base}, trip),
           "personal_full_same_recording": chunk_adapt("chrysler-full", full, trip)}
    print("chrysler full chunks", out["personal_full_same_recording"], flush=True)
    return out


def main() -> None:
    import argparse
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cars", type=int, default=None)
    ap.add_argument("--skip-ved", action="store_true")
    ap.add_argument("--drivetrain", default=None, help="model folder to evaluate (default: the app's)")
    ap.add_argument("--out", default="other_cars_evaluation.json")
    a = ap.parse_args()
    global DRIVETRAIN_DIR
    if a.drivetrain:
        DRIVETRAIN_DIR = ROOT / a.drivetrain
    full, dt_base = NormalBehaviorModel.load(NBM_DIR), NormalBehaviorModel.load(DRIVETRAIN_DIR)
    res = {"seed": SEED, "base_full": full.version, "base_drivetrain": dt_base.version,
           "injections": [list(x) for x in INJ]}
    res["ford"] = part_ford(full, dt_base)
    res["chrysler"] = part_chrysler(full, dt_base)
    if not a.skip_ved:
        res["ved"] = part_ved(a.max_cars)
        print(json.dumps(res["ved"]["summary"], indent=1))
        print(json.dumps(res["ved"]["consistency_by_type"], indent=1))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / a.out).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
