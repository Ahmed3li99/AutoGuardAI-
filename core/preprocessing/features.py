"""Derived features computed from actual timestamps. All are 'calculated
features' (not measurements) and are labelled as such in the UI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.schema import TIME_COL

# Signals the anomaly models consume when available, in this order.
MODEL_SIGNALS = ["rpm", "speed_kmh", "coolant_temp_c", "engine_load_pct", "throttle_pct",
                 "manifold_pressure_kpa", "intake_air_temp_c", "battery_voltage"]


def acceleration_mps2(df: pd.DataFrame) -> pd.Series:
    """d(speed)/dt using the real time column; NaN where dt <= 0 or speed missing."""
    if "speed_kmh" not in df:
        return pd.Series(np.nan, index=df.index)
    v = df["speed_kmh"].to_numpy(dtype=float) / 3.6
    t = df[TIME_COL].to_numpy(dtype=float)
    a = np.full(len(v), np.nan)
    if len(v) > 1:
        dt = np.diff(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            a[1:] = np.where(dt > 0, np.diff(v) / dt, np.nan)
    return pd.Series(a, index=df.index)


def rate_per_min(df: pd.DataFrame, col: str, window_s: int = 60) -> pd.Series:
    """Change of ``col`` over the trailing window, per minute (1 Hz input)."""
    if col not in df:
        return pd.Series(np.nan, index=df.index)
    return (df[col] - df[col].shift(window_s)) * (60.0 / window_s)


def window_features(df_1hz: pd.DataFrame, signals: list[str], window_s: int = 30) -> pd.DataFrame:
    """Per-second feature vector: value, rolling mean/std and first difference
    for each signal, plus acceleration. Used by the Isolation Forest baseline."""
    feats = {}
    for c in signals:
        s = df_1hz[c]
        feats[c] = s
        feats[f"{c}__mean{window_s}"] = s.rolling(window_s, min_periods=window_s // 2).mean()
        feats[f"{c}__std{window_s}"] = s.rolling(window_s, min_periods=window_s // 2).std()
        feats[f"{c}__diff"] = s.diff()
    if "speed_kmh" in signals:
        feats["accel_mps2"] = acceleration_mps2(df_1hz)
    return pd.DataFrame(feats, index=df_1hz.index)
