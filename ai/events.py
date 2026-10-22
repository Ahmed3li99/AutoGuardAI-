"""Alert/event structures shared by rules and models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from core.schema import TIME_COL

SEVERITIES = ("attention", "warning")


@dataclass
class Alert:
    t_start_s: float
    t_end_s: float
    severity: str                     # "attention" | "warning"
    detector: str                     # e.g. "rule:coolant_high", "model:isolation_forest"
    detector_kind: str                # "heuristic_rule" | "ml_anomaly"
    signals: list[str]
    explanation: str
    peak_score: float | None = None
    threshold: float | None = None
    model_version: str | None = None
    values: dict[str, float] = field(default_factory=dict)  # snapshot at peak
    origin: str = "recorded"          # copied from session origin

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectorScores:
    t_s: list[float]
    scores: list[float | None]          # None where data were insufficient
    threshold: dict | None = None


@dataclass
class AnalysisResult:
    alerts: list[Alert]
    scores: dict[str, DetectorScores]   # detector -> per-second scores
    detector_status: dict[str, dict]    # detector -> {status, reason, version, ...}
    timing_s: dict[str, float]
    config: dict

    def alerts_frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.to_dict() for a in self.alerts])


def segments(mask: np.ndarray, t: np.ndarray, min_duration_s: float = 0.0,
             merge_gap_s: float = 0.0) -> list[tuple[int, int]]:
    """Contiguous True runs as (start_idx, end_idx_inclusive), merged across
    gaps <= merge_gap_s and filtered by duration >= min_duration_s."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1 or (t[i] - t[prev]) <= merge_gap_s:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return [(a, b) for a, b in runs if t[b] - t[a] + 1 >= min_duration_s]


def snapshot(df: pd.DataFrame, i: int, signals: list[str]) -> dict[str, float]:
    row = df.iloc[i]
    return {c: round(float(row[c]), 3) for c in [TIME_COL, *signals]
            if c in df and pd.notna(row[c])}
