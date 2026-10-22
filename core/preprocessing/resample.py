"""Regular 1 Hz grid for windowed models.

carOBD logs ~2 rows per second but its clock (engine run time, PID 0x1F) has
1 s resolution, so sub-second ordering carries no timing information. We
therefore take the LAST reading within each whole second. Seconds with no
reading stay NaN unless they lie inside a short gap (<= ``ffill_limit_s``),
where the previous value is carried forward; longer gaps stay missing and are
masked by the models. The fraction of filled cells is reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.schema import CANONICAL_FIELDS, TIME_COL


@dataclass
class ResampleInfo:
    seconds: int
    filled_cells: int
    missing_cells_after: int


def resample_1hz(df: pd.DataFrame, signals: list[str] | None = None,
                 ffill_limit_s: int = 3) -> tuple[pd.DataFrame, ResampleInfo]:
    if signals is None:
        signals = [c for c in df.columns if c in CANONICAL_FIELDS and CANONICAL_FIELDS[c].numeric]
    if df.empty:
        return pd.DataFrame(columns=[TIME_COL, *signals]), ResampleInfo(0, 0, 0)
    sec = np.floor(df[TIME_COL].to_numpy(dtype=float)).astype(np.int64)
    g = df[signals].assign(_s=sec).groupby("_s", sort=True).last()
    full = pd.RangeIndex(int(sec.min()), int(sec.max()) + 1)
    g = g.reindex(full)
    before = int(g.isna().sum().sum())
    g = g.ffill(limit=ffill_limit_s)
    after = int(g.isna().sum().sum())
    out = g.reset_index(names=TIME_COL)
    out[TIME_COL] = out[TIME_COL].astype(float)
    return out, ResampleInfo(len(out), before - after, after)
