"""Explicit, documented anomaly injection - used ONLY for the synthetic demo
scenario and for software stress tests on held-out recorded trips. Results on
injected anomalies are never reported as real-world fault detection."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from core.schema import TIME_COL


@dataclass(frozen=True)
class InjectedEvent:
    kind: str
    signal: str
    start_s: float
    end_s: float
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


def _mask(df: pd.DataFrame, ev: InjectedEvent) -> pd.Series:
    return (df[TIME_COL] >= ev.start_s) & (df[TIME_COL] < ev.end_s)


def inject(df: pd.DataFrame, events: list[InjectedEvent]) -> pd.DataFrame:
    """Return a copy of ``df`` with the events applied. Supported kinds (fixed
    magnitudes): coolant_overheat, voltage_drop, flatline, spike_noise, dropout."""
    out = df.copy()
    for ev in events:
        if ev.signal not in out:
            continue
        m = _mask(out, ev)
        n = int(m.sum())
        if n == 0:
            continue
        s = out.loc[m, ev.signal].astype(float)
        if ev.kind == "coolant_overheat":
            # linear rise to 112 C, held: physically plausible overheating profile
            start = float(s.iloc[0]) if s.notna().any() else 90.0
            ramp = np.linspace(start, 112.0, max(n // 2, 1))
            vals = np.concatenate([ramp, np.full(n - ramp.size, 112.0)])
            out.loc[m, ev.signal] = vals
        elif ev.kind == "voltage_drop":
            out.loc[m, ev.signal] = 11.4 + 0.05 * np.sin(np.arange(n))
        elif ev.kind == "flatline":
            out.loc[m, ev.signal] = float(s.iloc[0]) if s.notna().any() else 0.0
        elif ev.kind == "spike_noise":
            rng = np.random.default_rng(7)
            scale = max(float(out[ev.signal].std(skipna=True) or 1.0), 1e-3)
            out.loc[m, ev.signal] = s + rng.normal(0, 3 * scale, n)
        elif ev.kind == "dropout":
            out.loc[m, ev.signal] = np.nan
        else:
            raise ValueError(f"Unknown injection kind {ev.kind}")
    return out


def label_vector(t_s: np.ndarray, events: list[InjectedEvent]) -> np.ndarray:
    y = np.zeros(len(t_s), dtype=bool)
    for ev in events:
        y |= (t_s >= ev.start_s) & (t_s < ev.end_s)
    return y
