"""Threshold calibration and score -> alert conversion for anomaly models.

Thresholds are quantiles of anomaly scores on the CALIBRATION partition
(normal-operation recorded trips, disjoint from training and test). The
quantile fixes the expected alert rate on data like the calibration trips;
it says nothing about recall on real faults (no labels exist)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ai.events import Alert, segments, snapshot
from core.schema import TIME_COL

ATTENTION_Q = 0.995
WARNING_Q = 0.9995


@dataclass
class Threshold:
    attention: float
    warning: float
    attention_quantile: float
    warning_quantile: float
    n_calibration_points: int
    calibration_source: str

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate(scores: np.ndarray, source: str, q_att: float = ATTENTION_Q,
              q_warn: float = WARNING_Q) -> Threshold:
    s = scores[np.isfinite(scores)]
    if s.size < 100:
        raise ValueError(f"Need >=100 finite calibration scores, got {s.size}")
    return Threshold(float(np.quantile(s, q_att)), float(np.quantile(s, q_warn)),
                     q_att, q_warn, int(s.size), source)


def scores_to_alerts(df_1hz: pd.DataFrame, scores: np.ndarray, thr: Threshold, detector: str,
                     model_version: str, contributions: pd.DataFrame | None = None,
                     origin: str = "recorded", min_duration_s: float = 5.0,
                     merge_gap_s: float = 10.0, note: str = "") -> list[Alert]:
    t = df_1hz[TIME_COL].to_numpy(dtype=float)
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    alerts = []
    for a, b in segments(finite > thr.attention, t, min_duration_s, merge_gap_s):
        seg = finite[a:b + 1]
        peak_i = a + int(np.argmax(seg))
        peak = float(finite[peak_i])
        sev = "warning" if peak > thr.warning else "attention"
        if contributions is not None and not contributions.empty:
            contrib = contributions.iloc[a:b + 1].mean().sort_values(ascending=False)
            top = [c for c in contrib.index[:2] if np.isfinite(contrib[c])]
        else:
            top = []
        sigs = top or [c for c in df_1hz.columns if c != TIME_COL][:2]
        expl = (f"Unusual multi-signal pattern for {t[b] - t[a] + 1:.0f} s "
                f"(peak score {peak:.3g} vs threshold {thr.attention:.3g}, "
                f"{thr.attention_quantile:.1%} calibration quantile). ")
        if top:
            expl += f"Largest deviation in: {', '.join(top)}. "
        expl += ("This is a statistical anomaly relative to the calibration trips, "
                 "not an identified mechanical fault." + (" " + note if note else ""))
        alerts.append(Alert(float(t[a]), float(t[b]), sev, detector, "ml_anomaly", sigs, expl,
                            peak, thr.attention, model_version, snapshot(df_1hz, peak_i, sigs), origin))
    return alerts
