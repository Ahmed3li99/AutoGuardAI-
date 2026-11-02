"""Expected-vs-actual ("normal behaviour") model - the explainable detector.

For each target signal, a gradient-boosted regressor learns from NORMAL train
trips what the signal usually is given the OTHER signals and the operating
context (engine run time, recent load/speed, ...). The target's own value is
never an input, so a frozen, drifting or jumping sensor cannot explain itself.

Per second and target:   residual = actual - expected
                         z = |residual| / robust scale (1.4826 * MAD on calibration trips)
An alert is raised when the 5-s rolling median of z exceeds the target's
threshold (a high quantile of the same quantity on calibration trips).
Every alert therefore carries: actual value, expected value, typical range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from ai.artifacts import load_artifact, save_artifact
from ai.events import Alert, segments, snapshot
from core.preprocessing.features import acceleration_mps2
from core.schema import TIME_COL

TARGETS = ["coolant_temp_c", "rpm", "battery_voltage", "manifold_pressure_kpa",
           "engine_load_pct", "intake_air_temp_c", "speed_kmh"]
CONTEXT_SIGNALS = ["rpm", "speed_kmh", "coolant_temp_c", "engine_load_pct", "throttle_pct",
                   "manifold_pressure_kpa", "intake_air_temp_c", "battery_voltage"]
SMOOTH_S = 5
MIN_DURATION_S = 8
MERGE_GAP_S = 10
THRESHOLD_Q = 0.999
SEED = 2024
RANGE_GATED = {"coolant_temp_c", "intake_air_temp_c"}
WARMUP_S = 900          # below-range temperatures are normal while the engine warms up
NO_TYPICAL_RANGE = {"speed_kmh"}   # a "normal speed" range is meaningless for explanations


def build_features(df: pd.DataFrame, context: list[str] | None = None) -> pd.DataFrame:
    """Context features on a 1 Hz frame. Missing signals become NaN columns
    (the regressors handle NaN natively). ``context`` selects the input signals
    (default: the full set; the drivetrain variant uses only RPM and speed)."""
    context = CONTEXT_SIGNALS if context is None else context
    f = pd.DataFrame(index=df.index)
    for c in context:
        f[c] = df[c] if c in df else np.nan
    f["run_time_s"] = (df[TIME_COL] - df[TIME_COL].iloc[0]).clip(upper=3600) if len(df) else []
    for c in ("engine_load_pct", "speed_kmh", "rpm"):
        if c not in context:
            continue
        f[f"{c}__mean60"] = f[c].rolling(60, min_periods=10).mean()
        f[f"{c}__mean300"] = f[c].rolling(300, min_periods=30).mean()
    f["accel_mps2"] = acceleration_mps2(df) if "speed_kmh" in df else np.nan
    f["accel_mps2"] = f["accel_mps2"].rolling(3, min_periods=1).mean()
    return f


def _inputs_for(target: str, feats: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in feats.columns if c == target or c.startswith(target + "__")]
    return feats.drop(columns=drop)


@dataclass
class TargetModel:
    target: str
    regressor: HistGradientBoostingRegressor
    input_columns: list[str]
    scale: float = 1.0                 # robust residual scale (calibration)
    threshold: float = np.inf          # on smoothed z (calibration quantile)
    typical: dict = field(default_factory=dict)   # train percentiles by state


@dataclass
class NormalBehaviorModel:
    models: dict[str, TargetModel]
    meta: dict

    @property
    def version(self) -> str:
        return self.meta.get("version", "nbm")

    @property
    def signals(self) -> list[str]:
        return list(self.models)

    @property
    def threshold(self):
        """Pipeline compatibility: thresholds are per target; an object with
        attention/warning normalised to 1.0 (score = z / target threshold)."""
        from ai.thresholds import Threshold
        return Threshold(1.0, 1.5, THRESHOLD_Q, THRESHOLD_Q, self.meta.get("n_calibration_points", 0),
                         self.meta.get("calibration_source", ""))

    def missing_signals(self, available: list[str]) -> list[str]:
        # Usable as long as at least one target and some context exist.
        return [] if any(t in available for t in self.models) else list(self.models)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Columns: <target>__expected, <target>__expected_base, <target>__z (smoothed).

        A personal (adapted) model carries ``car_map`` {signal: (a, b)} meaning
        this car's value ~= a + b * base-car value. Inputs are mapped INTO the base
        car's scale before the base regressors run, and the prediction is mapped back."""
        car_map = self.meta.get("car_map") or {}
        df_in = df
        if car_map:
            df_in = df.copy()
            for sig, (a, b) in car_map.items():
                if sig in df_in:
                    df_in[sig] = (df_in[sig] - a) / b
        feats = build_features(df_in, self.meta.get("context_signals"))
        out = pd.DataFrame(index=df.index)
        for t, m in self.models.items():
            if t not in df or df[t].isna().all():
                continue
            X = _inputs_for(t, feats)[m.input_columns]
            base_exp = m.regressor.predict(X.to_numpy(dtype=float))
            a, b = car_map.get(t, (0.0, 1.0))
            exp = a + b * base_exp
            out[f"{t}__expected_base"] = base_exp
            out[f"{t}__expected"] = exp
            z = (df[t] - exp).abs() / m.scale
            out[f"{t}__z"] = z.rolling(SMOOTH_S, min_periods=1, center=True).median()
        return out

    def score(self, df: pd.DataFrame, pred: pd.DataFrame | None = None) -> tuple[np.ndarray, pd.DataFrame]:
        """Overall per-second score = max over targets of z / target threshold
        (1.0 = at the alert threshold). Also returns the per-target ratios.
        Pass ``pred`` (from predict) to avoid recomputing the regressors."""
        pred = self.predict(df) if pred is None else pred
        ratios = pd.DataFrame({t: pred[f"{t}__z"] / self.models[t].threshold
                               for t in self.models if f"{t}__z" in pred}, index=df.index)
        if ratios.empty:
            return np.full(len(df), np.nan), ratios
        return ratios.max(axis=1, skipna=True).to_numpy(), ratios

    def alerts(self, df: pd.DataFrame, origin: str = "recorded", note: str = "",
               pred: pd.DataFrame | None = None) -> list[Alert]:
        pred = self.predict(df) if pred is None else pred
        t = df[TIME_COL].to_numpy(dtype=float)
        out = []
        for tgt, m in self.models.items():
            zc = f"{tgt}__z"
            if zc not in pred:
                continue
            z = pred[zc].to_numpy()
            mask = np.nan_to_num(z, nan=0.0) > m.threshold
            if tgt in RANGE_GATED and m.typical.get("all"):
                # Temperatures depend on start state (a warm restart looks "too hot"
                # early on), so they must also leave the car's normal warm range.
                lo, hi = m.typical["all"]
                mask &= ((df[tgt] > hi) | (df[tgt] < lo)).to_numpy()
                mask &= (df[TIME_COL] - df[TIME_COL].iloc[0] > WARMUP_S).to_numpy() | (df[tgt] > hi).to_numpy()
            for a, b in segments(mask, t, MIN_DURATION_S, MERGE_GAP_S):
                seg = slice(a, b + 1)
                peak = a + int(np.nanargmax(z[seg]))
                actual = float(df[tgt].iloc[peak])
                expected = float(pred[f"{tgt}__expected"].iloc[peak])
                ratio = float(z[peak] / m.threshold)
                sev = "warning" if ratio >= 1.5 else "attention"
                vals = snapshot(df, peak, [tgt])
                vals.update({f"{tgt}__expected": round(expected, 2),
                             f"{tgt}__seg_actual_max": round(float(df[tgt].iloc[seg].max()), 2),
                             f"{tgt}__seg_actual_min": round(float(df[tgt].iloc[seg].min()), 2),
                             f"{tgt}__seg_expected_mean": round(float(pred[f"{tgt}__expected"].iloc[seg].mean()), 2)})
                moving = "speed_kmh" not in df or not (df["speed_kmh"].iloc[peak] < 2)
                rng = m.typical.get("moving" if moving else "idle") or m.typical.get("all")
                if tgt in RANGE_GATED:
                    rng = m.typical.get("all")
                if rng and tgt not in NO_TYPICAL_RANGE:
                    vals[f"{tgt}__typical_low"], vals[f"{tgt}__typical_high"] = rng
                expl = (f"{tgt}: measured {actual:.1f}, expected about {expected:.1f} for these driving "
                        f"conditions (deviation {ratio:.1f}x the normal limit) for {t[b] - t[a] + 1:.0f} s. "
                        "Expected value learned from normal trips of the calibration vehicle; "
                        "not a diagnosis." + (" " + note if note else ""))
                out.append(Alert(float(t[a]), float(t[b]), sev, "model:expected_vs_actual", "ml_anomaly",
                                 [tgt], expl, ratio, 1.0, self.version, vals, origin))
        return out

    def save(self, directory: Path) -> None:
        save_artifact(self.models, self.meta, directory, "nbm")

    @classmethod
    def load(cls, directory: Path) -> "NormalBehaviorModel":
        models, meta = load_artifact(directory, "nbm")
        if meta.get("sklearn_version") != sklearn.__version__:
            meta["load_warning"] = f"Trained with scikit-learn {meta.get('sklearn_version')}"
        return cls(models, meta)


def typical_ranges(frames: list[pd.DataFrame], target: str) -> dict:
    allf = pd.concat([f for f in frames if target in f], ignore_index=True)
    if allf.empty:
        return {}
    # typical ranges describe a WARM engine (coolant >= 75 C) where available
    warm = allf[allf["coolant_temp_c"] >= 75] if "coolant_temp_c" in allf else allf
    out = {}
    for name, sub in (("all", warm),
                      ("idle", warm[warm["speed_kmh"] < 2] if "speed_kmh" in warm else warm.iloc[0:0]),
                      ("moving", warm[warm["speed_kmh"] >= 2] if "speed_kmh" in warm else warm.iloc[0:0])):
        v = sub[target].dropna()
        if len(v) > 200:
            out[name] = [round(float(v.quantile(0.025)), 1), round(float(v.quantile(0.975)), 1)]
    return out


SMALL_CALIBRATION = 3000      # seconds; below this a 99.9 % quantile is just the largest point
SMALL_MARGIN = 1.10           # small calibration sets: threshold = 110 % of the largest calibration z


def fit(train: list[pd.DataFrame], calib: list[pd.DataFrame], meta: dict,
        min_rows: int = 1000, context: list[str] | None = None,
        targets: list[str] | None = None) -> NormalBehaviorModel:
    context = CONTEXT_SIGNALS if context is None else context
    targets = TARGETS if targets is None else targets
    feats_tr = [build_features(f, context) for f in train]
    models: dict[str, TargetModel] = {}
    n_cal = 0
    for tgt in targets:
        X = pd.concat([_inputs_for(tgt, ft) for ft, f in zip(feats_tr, train) if tgt in f], ignore_index=True)
        y = pd.concat([f[tgt] for f in train if tgt in f], ignore_index=True)
        ok = y.notna().to_numpy()
        if ok.sum() < min_rows:
            continue
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                            l2_regularization=1.0, random_state=SEED)
        reg.fit(X.to_numpy(dtype=float)[ok], y.to_numpy(dtype=float)[ok])
        tm = TargetModel(tgt, reg, list(X.columns), typical=typical_ranges(train, tgt))
        # calibration: robust residual scale, then threshold on smoothed z
        res, zs = [], []
        for f in calib:
            if tgt not in f:
                continue
            Xc = _inputs_for(tgt, build_features(f, context))[tm.input_columns]
            r = f[tgt].to_numpy(dtype=float) - reg.predict(Xc.to_numpy(dtype=float))
            res.append(r)
        r_all = np.concatenate(res)
        r_all = r_all[np.isfinite(r_all)]
        mad = float(np.median(np.abs(r_all - np.median(r_all))))
        tm.scale = max(1.4826 * mad, 1e-3)
        for f, r in zip([f for f in calib if tgt in f], res):
            z = pd.Series(np.abs(r) / tm.scale).rolling(SMOOTH_S, min_periods=1, center=True).median()
            zs.append(z.to_numpy())
        z_all = np.concatenate(zs)
        z_all = z_all[np.isfinite(z_all)]
        if z_all.size >= SMALL_CALIBRATION:
            tm.threshold = float(np.quantile(z_all, THRESHOLD_Q))
        else:   # few calibration seconds (personal model): conservative, never fires on its own calibration trips
            tm.threshold = float(np.max(z_all) * SMALL_MARGIN) if z_all.size else np.inf
        n_cal = max(n_cal, int(z_all.size))
        models[tgt] = tm
    meta = {**meta, "targets": list(models), "context_signals": list(context), "threshold_quantile": THRESHOLD_Q, "smooth_s": SMOOTH_S,
            "threshold_rule": f"q{THRESHOLD_Q} if >= {SMALL_CALIBRATION} calibration s else {SMALL_MARGIN} x max",
            "n_calibration_points": n_cal, "sklearn_version": sklearn.__version__, "seed": SEED,
            "per_target": {t: {"scale": m.scale, "threshold_z": m.threshold, "typical": m.typical}
                           for t, m in models.items()}}
    return NormalBehaviorModel(models, meta)
