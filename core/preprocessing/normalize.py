"""Raw table + ColumnMapping -> NormalizedSession, with a full quality account."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.mapping import ColumnMapping, convert_units, parse_time_column
from core.schema import (CANONICAL_FIELDS, TIME_COL, DataOrigin, NormalizedSession,
                         SessionMetadata, TimeBasis)
from core.validation.quality import QualityReport, build_quality_report


def normalize(raw: pd.DataFrame, mapping: ColumnMapping, source_name: str,
              origin: DataOrigin = DataOrigin.RECORDED, source_sha256: str | None = None,
              playback_interval_s: float | None = None) -> tuple[NormalizedSession, QualityReport]:
    if raw.empty:
        raise ValueError("The file contains no data rows.")
    rules = mapping.resolved_columns(list(raw.columns))
    if not rules:
        raise ValueError("None of the file's columns map to a known vehicle signal. "
                         "Use manual column mapping.")

    out = pd.DataFrame(index=raw.index)
    parse_failures: dict[str, int] = {}
    invalid: dict[str, int] = {}
    conversions: dict[str, str] = {}
    notes: list[str] = []

    # --- time -----------------------------------------------------------
    start_ts = None
    basis = mapping.time_basis
    interval = None
    if mapping.time_column and mapping.time_column in raw.columns:
        t, start_ts, bad_t = parse_time_column(raw[mapping.time_column], basis, mapping.time_unit)
        if bad_t:
            parse_failures[TIME_COL] = bad_t
        if np.all(np.isnan(t)):
            raise ValueError(f"Time column '{mapping.time_column}' could not be parsed.")
        out[TIME_COL] = t
    else:
        if mapping.time_column:
            notes.append(f"Configured time column '{mapping.time_column}' not found.")
        basis = TimeBasis.SYNTHETIC_CLOCK
        interval = playback_interval_s or mapping.playback_interval_s
        out[TIME_COL] = np.arange(len(raw), dtype=float) * interval
        notes.append(f"No time in source: synthetic playback clock at {interval:g} s per row "
                     "(not a recorded time).")

    # --- signals -------------------------------------------------------------
    for src, rule in rules.items():
        spec = CANONICAL_FIELDS[rule.to]
        if not spec.numeric:
            out[rule.to] = raw[src].astype("string")
            continue
        s = raw[src]
        num = pd.to_numeric(s, errors="coerce")
        nonblank = s.notna() & (s.astype(str).str.strip() != "")
        n_bad = int((num.isna() & nonblank).sum())
        if n_bad:
            parse_failures[rule.to] = n_bad
        num, desc = convert_units(num.astype(float), rule.unit, rule.to)
        if desc:
            conversions[rule.to] = desc + (" (unit inferred)" if rule.unit_inferred else "")
        lo, hi = spec.valid_min, spec.valid_max
        bad_mask = pd.Series(False, index=num.index)
        if lo is not None:
            bad_mask |= num < lo - 1e-9
        if hi is not None:
            bad_mask |= num > hi + 1e-9
        if bad_mask.any():
            invalid[rule.to] = int(bad_mask.sum())
            num = num.mask(bad_mask)  # impossible values -> missing (counted in report)
        out[rule.to] = num

    # Respect an explicit data_origin column (e.g. a re-imported synthetic export).
    if "data_origin" in raw.columns:
        vals = set(raw["data_origin"].dropna().astype(str).str.lower())
        if vals == {"synthetic"}:
            origin = DataOrigin.SYNTHETIC
            notes.append("data_origin column marks this file as SYNTHETIC.")

    n_nonmono = int((np.diff(out[TIME_COL].to_numpy()) < 0).sum())
    n_time_nan = int(out[TIME_COL].isna().sum())
    if n_time_nan:
        out = out[out[TIME_COL].notna()]
        notes.append(f"Dropped {n_time_nan} rows without a parseable time value.")
    if n_nonmono:
        notes.append(f"Time went backwards {n_nonmono} times; rows were stably sorted by time.")
        out = out.sort_values(TIME_COL, kind="stable")
    out = out.reset_index(drop=True)

    meta = SessionMetadata(
        source_name=source_name, origin=origin, time_basis=basis,
        dataset_id=mapping.dataset_id, mapping_id=mapping.mapping_id,
        source_sha256=source_sha256, start_timestamp=start_ts,
        playback_interval_s=interval,
        column_map={k: v.to for k, v in rules.items()},
        unit_conversions=conversions, notes=notes,
    )
    session = NormalizedSession(out, meta)
    unmapped = [c for c in raw.columns if c not in rules and c != mapping.time_column]
    report = build_quality_report(session, raw_rows=len(raw), parse_failures=parse_failures,
                                  invalid=invalid, non_monotonic=n_nonmono, unmapped=unmapped)
    return session, report
