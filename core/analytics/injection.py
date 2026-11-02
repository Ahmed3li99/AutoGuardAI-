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
    magnitude: float | None = None     # only for kinds that take one (step_jump, voltage_decline)

    def to_dict(self) -> dict:
        return asdict(self)


def _mask(df: pd.DataFrame, ev: InjectedEvent) -> pd.Series:
    return (df[TIME_COL] >= ev.start_s) & (df[TIME_COL] < ev.end_s)


def inject(df: pd.DataFrame, events: list[InjectedEvent]) -> pd.DataFrame:
    """Return a copy of ``df`` with the events applied. Kinds: coolant_overheat,
    voltage_drop, flatline, spike_noise, dropout (fixed magnitudes), voltage_decline
    (gradual fall to ``magnitude`` volts, default 11.8) and step_jump (sudden offset of
    ``magnitude`` in the signal's unit, default +20)."""
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
            # physically plausible profile: rise to 112 C (1/2), hold (1/4), cool back down (1/4)
            # to the recorded value. An instant drop at the end would itself be an impossible
            # 1-second jump that the sensor-consistency check (rightly) flags.
            start = float(s.iloc[0]) if s.notna().any() else 90.0
            end = float(s.iloc[-1]) if s.notna().any() else start
            n_up, n_hold = max(n // 2, 1), max(n // 4, 0)
            n_down = n - n_up - n_hold
            vals = np.concatenate([np.linspace(start, 112.0, n_up), np.full(n_hold, 112.0),
                                   np.linspace(112.0, end, n_down + 1)[1:]])
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
        elif ev.kind == "voltage_decline":
            # slowly weakening charging: linear fall from the current level to the target
            start = float(s.iloc[0]) if s.notna().any() else 13.8
            out.loc[m, ev.signal] = np.linspace(start, ev.magnitude if ev.magnitude is not None else 11.8, n)
        elif ev.kind == "step_jump":
            # a sensor suddenly reading off by a constant offset (glitch / calibration fault)
            out.loc[m, ev.signal] = s + (ev.magnitude if ev.magnitude is not None else 20.0)
        else:
            raise ValueError(f"Unknown injection kind {ev.kind}")
    return out


def label_vector(t_s: np.ndarray, events: list[InjectedEvent]) -> np.ndarray:
    y = np.zeros(len(t_s), dtype=bool)
    for ev in events:
        y |= (t_s >= ev.start_s) & (t_s < ev.end_s)
    return y


def changes_data(original: pd.DataFrame, injected: pd.DataFrame, ev: InjectedEvent, tol: float = 1e-6) -> bool:
    """Label check: an injected 'fault' only counts if it really changed the signal in its window.
    Freezing a reading that was already constant (e.g. a parked car at steady idle) is not a fault,
    and counting it would make every detector look worse (or, for the inverse case, better) than it is."""
    m = (original[TIME_COL] >= ev.start_s) & (original[TIME_COL] < ev.end_s)
    if ev.signal not in original or not m.any():
        return False
    a = original.loc[m, ev.signal].to_numpy(float)
    b = injected.loc[m, ev.signal].to_numpy(float)
    both = np.isfinite(a) & np.isfinite(b)
    changed_values = bool(both.any() and np.nanmax(np.abs(a[both] - b[both])) > tol)
    changed_missing = bool((np.isfinite(a) != np.isfinite(b)).any())
    return changed_values or changed_missing
