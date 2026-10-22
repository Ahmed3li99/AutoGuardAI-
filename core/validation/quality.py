"""Data-quality report with actual counts (no estimated or invented numbers)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from core.schema import CANONICAL_FIELDS, TIME_COL, NormalizedSession, TimeBasis


@dataclass
class SignalQuality:
    name: str
    unit: str
    present: int
    missing: int
    missing_rate: float
    invalid_out_of_range: int
    parse_failures: int
    min: float | None
    median: float | None
    max: float | None


@dataclass
class QualityReport:
    raw_rows: int
    rows: int
    time_basis: str
    duration_s: float
    median_dt_s: float | None
    rows_per_second: float | None
    n_gaps: int                      # dt > max(3 * median dt, 2 s)
    max_gap_s: float | None
    non_monotonic_time: int
    duplicate_rows: int
    duplicate_time_values: int
    signals: list[SignalQuality] = field(default_factory=list)
    unmapped_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.rows == 0 or not self.signals:
            return "Unusable"
        if any(w.startswith("[ERROR]") for w in self.warnings):
            return "Unusable"
        return "Warnings" if self.warnings else "OK"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        return d


def build_quality_report(session: NormalizedSession, raw_rows: int, parse_failures: dict[str, int],
                         invalid: dict[str, int], non_monotonic: int,
                         unmapped: list[str]) -> QualityReport:
    df = session.data
    t = df[TIME_COL].to_numpy(dtype=float)
    dt = np.diff(t)
    pos = dt[dt > 0]
    median_dt = float(np.median(pos)) if pos.size else None
    gap_thr = max(3 * median_dt, 2.0) if median_dt else None
    gaps = dt[dt > gap_thr] if gap_thr else np.array([])
    duration = float(t[-1] - t[0]) if t.size else 0.0
    sigs = []
    warnings: list[str] = []
    for c in df.columns:
        if c == TIME_COL or c not in CANONICAL_FIELDS or not CANONICAL_FIELDS[c].numeric:
            continue
        s = df[c]
        n_miss = int(s.isna().sum())
        valid = s.dropna()
        sigs.append(SignalQuality(
            name=c, unit=CANONICAL_FIELDS[c].unit, present=int(valid.size), missing=n_miss,
            missing_rate=round(n_miss / len(s), 4) if len(s) else 1.0,
            invalid_out_of_range=invalid.get(c, 0), parse_failures=parse_failures.get(c, 0),
            min=float(valid.min()) if valid.size else None,
            median=float(valid.median()) if valid.size else None,
            max=float(valid.max()) if valid.size else None))
        if valid.size == 0:
            warnings.append(f"{c}: mapped but contains no valid values (shown as N/A).")
        elif n_miss / len(s) > 0.2:
            warnings.append(f"{c}: {n_miss / len(s):.0%} of values missing.")
        if invalid.get(c):
            warnings.append(f"{c}: {invalid[c]} physically impossible values treated as missing.")
        if parse_failures.get(c):
            warnings.append(f"{c}: {parse_failures[c]} non-numeric values treated as missing.")
    if session.meta.time_basis == TimeBasis.SYNTHETIC_CLOCK:
        warnings.append("No recorded time: replay uses a synthetic playback clock.")
    if non_monotonic:
        warnings.append(f"Time decreased {non_monotonic} times in the source.")
    if gaps.size:
        warnings.append(f"{gaps.size} time gaps longer than {gap_thr:.1f} s (max {gaps.max():.1f} s).")
    if len(df) < 30:
        warnings.append("Fewer than 30 rows: too short for windowed anomaly models.")
    for u, desc in session.meta.unit_conversions.items():
        if "inferred" in desc:
            warnings.append(f"{u}: unit conversion {desc} - verify before trusting values.")
    if not sigs:
        warnings.append("[ERROR] No numeric vehicle signal could be mapped.")
    dup_time = int(df[TIME_COL].duplicated().sum())
    return QualityReport(
        raw_rows=raw_rows, rows=len(df), time_basis=session.meta.time_basis.value,
        duration_s=round(duration, 3), median_dt_s=median_dt,
        rows_per_second=round(len(df) / duration, 3) if duration > 0 else None,
        n_gaps=int(gaps.size), max_gap_s=float(gaps.max()) if gaps.size else None,
        non_monotonic_time=non_monotonic,
        duplicate_rows=int(df.duplicated().sum()), duplicate_time_values=dup_time,
        signals=sigs, unmapped_columns=list(unmapped), warnings=warnings)
