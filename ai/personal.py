"""'Learn your car': a personal expected-vs-actual model per vehicle profile.

The user approves REFERENCE trips (normal driving, no confirmed warning). From
at least MIN_TRIPS trips and MIN_MINUTES of driving, the BASE expected-vs-actual
model is ADAPTED to the car - it is not refitted from scratch.

Why adapt instead of refit: an experiment (experiments/evaluate_personal.py)
showed that regressors fitted from scratch on 3-5 trips raise 4-18 false
alerts/h on new trips of the same car (base: 0.57/h), because a few trips cannot
cover the variety of driving situations the base model saw in 77 trips.
Adaptation keeps the base model's knowledge and learns only a robust linear map
per signal, this car's value ~= a + b * base-car value (e.g. a hotter thermostat
= offset, different gearing = RPM factor):
  * EVIDENCE GATE: a signal is adapted only if its median offset from the base
    expectation is >= GATE_K x the base noise scale; otherwise it is left exactly
    as in the base model (so a car like the base car changes almost nothing);
  * a, b from a Huber regression of actual vs base-expected on the older
    reference trips (2 passes, inputs mapped into base scale), b clipped to 0.5-2;
    temperatures and voltage get an offset only (b = 1): physically a level shift;
  * residual scale re-estimated on the newest reference trips (never the same trips):
    adapted signals 0.8-3 x the base scale, the others 1-3 x (only ever less sensitive);
  * typical ranges = base ranges mapped through a + b x (unless 3 h of own data);
  * the z threshold stays the base model's (calibrated on 33,000+ s).
SCALE_CLIP, OWN_RANGE_MIN_S and GATE_K were chosen on the CALIBRATION partition
(never on test trips); NOISE_UP_TO on VED dev cars; see docs/EVALUATION.md §F, §G.

Safeguards against learning a problem as "normal":
  * a trip with a rule warning cannot be marked as reference (repository check);
  * reference trips with data-consistency findings are reported back;
  * the personal model is only used for trips assigned to that car.
"""
from __future__ import annotations

import copy
import datetime as dt
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from ai import normal_behavior as nbm
from ai.normal_behavior import NormalBehaviorModel
from core.paths import user_data_dir
from core.preprocessing.resample import resample_1hz

MIN_TRIPS = 3
MIN_MINUTES = 30.0
RECOMMENDED_TRIPS = 5           # more reference trips -> steadier map and noise margin (EVALUATION §F)
MOVING_KMH = 20.0
MIN_MOVING_MINUTES = 10.0       # learning trips must include real driving, not only idling
SCALE_CLIP = (0.8, 1.2)         # adapted residual scale relative to the base scale
# Every signal also learns this car's own residual spread, from 1x up to NOISE_UP_TO x the base scale
# (never narrower than the base). Real other cars (VED) spread 2-6x wider than the Toyota; chosen on
# VED DEV cars by a rule fixed beforehand (experiments/select_cross_car_settings.py, EVALUATION §G).
NOISE_UP_TO = 3.0
OWN_RANGE_MIN_S = 3 * 3600      # warm seconds needed to use the car's own typical range (else mapped)
GATE_K = 2.0                    # adapt a signal only if |median offset| >= GATE_K x base noise scale
OFFSET_ONLY = {"coolant_temp_c", "intake_air_temp_c", "battery_voltage"}   # level shifts, not factors


def vehicle_dir(vehicle_id: int) -> Path:
    return user_data_dir() / "vehicles" / str(vehicle_id)


def has_personal_model(vehicle_id: int | None) -> bool:
    return vehicle_id is not None and (vehicle_dir(vehicle_id) / "nbm_meta.json").exists()


def load_personal_model(vehicle_id: int) -> NormalBehaviorModel:
    return NormalBehaviorModel.load(vehicle_dir(vehicle_id))


def moving_minutes(frames: list[pd.DataFrame]) -> float:
    """Minutes above MOVING_KMH on a 1 Hz grid (unknown speed counts as not moving)."""
    return float(sum(int((f["speed_kmh"] > MOVING_KMH).sum()) for f in frames if "speed_kmh" in f) / 60.0)


def readiness(repo, vehicle_id: int) -> dict:
    """Ready when there are enough reference trips/minutes AND the trips used for
    learning contain real driving. Measured: learning only from parked/idle trips
    produced 20 false alerts/h on a simulated other car (docs/EVALUATION.md)."""
    ids = repo.reference_sessions(vehicle_id)
    df = repo.list_sessions(vehicle_id)
    minutes = float(df[df["id"].isin(ids)]["duration_s"].sum() / 60.0) if len(df) else 0.0
    enough = len(ids) >= MIN_TRIPS and minutes >= MIN_MINUTES
    train_moving = 0.0
    if enough:
        train_ids, _ = split_reference(ids)
        train_moving = moving_minutes(_frames(repo, train_ids))
    driving_ok = train_moving >= MIN_MOVING_MINUTES
    return {"trips": len(ids), "minutes": round(minutes, 1), "need_trips": MIN_TRIPS, "need_minutes": MIN_MINUTES,
            "recommended_trips": RECOMMENDED_TRIPS, "train_moving_minutes": round(train_moving, 1),
            "need_moving_minutes": MIN_MOVING_MINUTES, "driving_ok": driving_ok if enough else None,
            "ready": enough and driving_ok,
            "progress": min(1.0, min(len(ids) / MIN_TRIPS, minutes / MIN_MINUTES))}


def _frames(repo, ids: list[int]) -> list[pd.DataFrame]:
    out = []
    for sid in ids:
        s, _ = repo.load_session(sid)
        df, _ = resample_1hz(s.data, s.signals)
        out.append(df)
    return out


def split_reference(ids: list[int]) -> tuple[list[int], list[int]]:
    """Older trips train, newest ~1/3 (at least one) calibrate - never the same trip."""
    ids = sorted(ids)
    n_cal = max(1, math.ceil(len(ids) / 3))
    return ids[:-n_cal], ids[-n_cal:]


def adapt_base_model(base: NormalBehaviorModel, train: list[pd.DataFrame], calib: list[pd.DataFrame],
                     meta: dict, min_points: int = 300, noise_up_to: float | None = None) -> NormalBehaviorModel:
    """Adapt the base model to one car (see module docstring)."""
    model = copy.deepcopy(base)
    noise_up_to = NOISE_UP_TO if noise_up_to is None else noise_up_to
    model.meta = {**base.meta, **meta, "adapted_from": base.version, "noise_up_to": noise_up_to}
    car_map: dict[str, tuple[float, float]] = {}
    base_scale = {t: m.scale for t, m in base.models.items()}
    gated: set[str] | None = None
    for _ in range(2):                                     # pass 2 uses inputs mapped by pass 1
        model.meta["car_map"] = dict(car_map)
        preds = [model.predict(f) for f in train]
        new = {}
        for t in model.models:
            if gated is not None and t not in gated:
                continue
            xs = [p[f"{t}__expected_base"] for p in preds if f"{t}__expected_base" in p]
            ys = [f[t] for f, p in zip(train, preds) if f"{t}__expected_base" in p]
            if not xs:
                continue
            x, y = pd.concat(xs).to_numpy(float), pd.concat(ys).to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < min_points or np.std(x[ok]) < 1e-6:
                continue
            if gated is None and abs(float(np.median(y[ok] - x[ok]))) < GATE_K * base_scale[t]:
                continue      # no clear systematic difference for this signal: keep the base model as is
            if t in OFFSET_ONLY:
                # physically a level shift (thermostat / alternator set-point), not a factor; a slope
                # fitted across cold start -> warm engine is meaningless (seen: x0.5 + 49 for a +6 C car)
                b = 1.0
            else:
                h = HuberRegressor(max_iter=500).fit(x[ok].reshape(-1, 1), y[ok])
                b = float(np.clip(h.coef_[0], 0.5, 2.0))
            a = float(np.median(y[ok] - b * x[ok]))         # robust intercept
            new[t] = (a, b)
        if gated is None:
            gated = set(new)
        car_map = new
    model.meta["car_map"] = {k: [round(a, 4), round(b, 4)] for k, (a, b) in car_map.items()}
    # residual scale on the newest reference trips (never used for a, b)
    pc = [model.predict(f) for f in calib]
    for t, m in model.models.items():
        r = [(f[t] - p[f"{t}__expected"]).to_numpy(float) for f, p in zip(calib, pc) if f"{t}__expected" in p]
        r = np.concatenate(r) if r else np.array([])
        r = r[np.isfinite(r)]
        if r.size >= 60:
            mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
            s0 = base_scale[t]
            if t in car_map:
                m.scale = float(np.clip(mad, SCALE_CLIP[0] * s0, max(SCALE_CLIP[1], noise_up_to or 0) * s0))
            elif noise_up_to:
                # this car is just less predictable (other gearing/gearbox): widen, never narrow, the margin
                m.scale = float(np.clip(mad, s0, noise_up_to * s0))
        if t not in car_map:
            continue
        a_t, b_t = car_map[t]
        allf = pd.concat([f for f in train + calib if t in f], ignore_index=True)
        warm = allf[allf["coolant_temp_c"] >= 75] if "coolant_temp_c" in allf else allf
        if warm[t].notna().sum() >= OWN_RANGE_MIN_S:      # enough of this car's own data
            m.typical = nbm.typical_ranges(train + calib, t)
        else:                                             # otherwise: base range mapped onto this car
            m.typical = {k: [round(a_t + b_t * lo, 1), round(a_t + b_t * hi, 1)] for k, (lo, hi) in m.typical.items()}
    return model


def train_personal_model(repo, vehicle_id: int) -> dict:
    """Fit + calibrate the personal model; returns a summary stored on the vehicle."""
    r = readiness(repo, vehicle_id)
    if not r["ready"]:
        if r["driving_ok"] is False:
            raise ValueError(f"The learning trips contain only {r['train_moving_minutes']} min of driving above "
                             f"{MOVING_KMH:.0f} km/h (need {MIN_MOVING_MINUTES:.0f}). Add a trip with real driving.")
        raise ValueError(f"Need at least {MIN_TRIPS} reference trips and {MIN_MINUTES:.0f} min "
                         f"(have {r['trips']} trips, {r['minutes']} min).")
    ids = repo.reference_sessions(vehicle_id)
    train_ids, cal_ids = split_reference(ids)
    train, calib = _frames(repo, train_ids), _frames(repo, cal_ids)
    today = dt.datetime.now().isoformat(timespec="seconds")
    from ai.model_registry import NBM_DIR, NBM_DRIVETRAIN_DIR, choose_variant
    # same rule as the analysis: a car whose logs lack most sensors adapts the drivetrain model
    present = set.intersection(*[{c for c in f.columns if f[c].notna().any()} for f in train + calib])
    variant = choose_variant(sorted(present))
    base = NormalBehaviorModel.load(NBM_DIR if variant == "full" else NBM_DRIVETRAIN_DIR)
    model = adapt_base_model(base, train, calib, {
        "version": f"personal-vehicle{vehicle_id}-{today[:10]}", "scope": "vehicle", "vehicle_id": vehicle_id,
        "variant": variant,
        "trained_on": f"vehicle {vehicle_id}: base model adapted with {len(train_ids)} reference trips",
        "train_session_ids": train_ids, "calibration_session_ids": cal_ids,
        "calibrated_on_dataset": f"vehicle:{vehicle_id}",
        "calibration_source": f"vehicle {vehicle_id}: {len(cal_ids)} reference trips"})
    out = vehicle_dir(vehicle_id)
    if out.exists():
        shutil.rmtree(out)
    model.save(out)
    # self-check: base vs personal alerts per hour on the calibration trips (unseen when fitting a, b)
    hours = sum(len(f) for f in calib) / 3600
    base_n = sum(len(base.alerts(f)) for f in calib)
    pers_n = sum(len(model.alerts(f)) for f in calib)
    info = {"trained_at": today, "version": model.version, "variant": variant, "targets": list(model.models),
            "train_trips": len(train_ids), "calibration_trips": len(cal_ids),
            "train_minutes": round(sum(len(f) for f in train) / 60, 1),
            "calibration_minutes": round(hours * 60, 1),
            "base_alerts_per_hour_on_calibration": round(base_n / hours, 2) if hours else None,
            "personal_alerts_per_hour_on_calibration": round(pers_n / hours, 2) if hours else None,
            "typical": {t: m.typical.get("all") for t, m in model.models.items()},
            "car_map": model.meta.get("car_map", {})}
    repo.set_vehicle_model(vehicle_id, info)
    return info


def forget_personal_model(repo, vehicle_id: int) -> None:
    if vehicle_dir(vehicle_id).exists():
        shutil.rmtree(vehicle_dir(vehicle_id))
    repo.set_vehicle_model(vehicle_id, None)
