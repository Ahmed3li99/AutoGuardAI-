"""Source-column -> canonical-field mapping with explicit unit conversion."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from core.paths import MAPPINGS_DIR
from core.schema import CANONICAL_FIELDS, TimeBasis

# Conversions to the canonical unit of each field. Keys are (from_unit, canonical_unit).
_CONVERSIONS: dict[tuple[str, str], Callable[[pd.Series], pd.Series]] = {
    ("degF", "°C"): lambda s: (s - 32.0) * 5.0 / 9.0,
    ("K", "°C"): lambda s: s - 273.15,
    ("mph", "km/h"): lambda s: s * 1.609344,
    ("psi", "kPa"): lambda s: s * 6.894757,
    ("MPa", "kPa"): lambda s: s * 1000.0,
    ("bar", "kPa"): lambda s: s * 100.0,
    ("kg/h", "g/s"): lambda s: s / 3.6,
    ("lb/min", "g/s"): lambda s: s * 453.59237 / 60.0,
    ("mV", "V"): lambda s: s / 1000.0,
    ("inHg", "kPa"): lambda s: s * 3.386389,
}
# Spellings that already equal the canonical unit.
_IDENTITY = {"degC": "°C", "C": "°C", "°C": "°C", "km/h": "km/h", "kph": "km/h",
             "kPa": "kPa", "rpm": "rpm", "%": "%", "V": "V", "g/s": "g/s",
             "deg": "°", "°": "°"}

UNITS_BY_FIELD: dict[str, list[str]] = {
    f: sorted({u for (u, c) in _CONVERSIONS if c == spec.unit} | {spec.unit})
    for f, spec in CANONICAL_FIELDS.items() if spec.numeric
}


def convert_units(series: pd.Series, from_unit: str, field_name: str) -> tuple[pd.Series, str | None]:
    """Return (converted series, description or None if no conversion was needed)."""
    target = CANONICAL_FIELDS[field_name].unit
    if from_unit == target or _IDENTITY.get(from_unit, from_unit) == target:
        return series, None
    fn = _CONVERSIONS.get((from_unit, target))
    if fn is None:
        raise ValueError(f"No conversion from '{from_unit}' to '{target}' for {field_name}")
    return fn(series), f"{from_unit} -> {target}"


def clean_column_name(name: object) -> str:
    """Strip whitespace and an empty unit suffix like ' ()' used by carOBD."""
    s = str(name).strip()
    s = re.sub(r"\s*\(\s*\)\s*$", "", s)
    return s


@dataclass
class ColumnRule:
    to: str
    unit: str
    unit_inferred: bool = False


@dataclass
class ColumnMapping:
    mapping_id: str
    dataset_id: str | None
    description: str
    detect_columns: list[str]
    time_column: str | None
    time_basis: TimeBasis
    time_unit: str = "s"
    playback_interval_s: float = 1.0
    columns: dict[str, ColumnRule] = field(default_factory=dict)
    canonical_passthrough: bool = False

    @classmethod
    def from_json(cls, d: dict) -> "ColumnMapping":
        t = d.get("time")
        cols = d.get("columns", {})
        passthrough = cols == "canonical"
        return cls(
            mapping_id=d["mapping_id"],
            dataset_id=d.get("dataset_id"),
            description=d.get("description", ""),
            detect_columns=d.get("detect_columns", []),
            time_column=t["column"] if t else None,
            time_basis=TimeBasis(t["basis"]) if t else TimeBasis.SYNTHETIC_CLOCK,
            time_unit=t.get("unit", "s") if t else "s",
            playback_interval_s=float(d.get("default_playback_interval_s", 1.0)),
            columns={} if passthrough else {k: ColumnRule(**v) for k, v in cols.items()},
            canonical_passthrough=passthrough,
        )

    def to_json(self) -> dict:
        return {
            "mapping_id": self.mapping_id,
            "dataset_id": self.dataset_id,
            "description": self.description,
            "detect_columns": self.detect_columns,
            "time": None if self.time_column is None else
                {"column": self.time_column, "basis": self.time_basis.value, "unit": self.time_unit},
            "default_playback_interval_s": self.playback_interval_s,
            "columns": "canonical" if self.canonical_passthrough else
                {k: {"to": r.to, "unit": r.unit, "unit_inferred": r.unit_inferred}
                 for k, r in self.columns.items()},
        }

    def resolved_columns(self, source_columns: list[str]) -> dict[str, ColumnRule]:
        if self.canonical_passthrough:
            return {c: ColumnRule(c, CANONICAL_FIELDS[c].unit) for c in source_columns
                    if c in CANONICAL_FIELDS}
        return {k: v for k, v in self.columns.items() if k in source_columns}


def load_builtin_mappings(directory: Path = MAPPINGS_DIR) -> list[ColumnMapping]:
    out = []
    for p in sorted(directory.glob("*.json")):
        out.append(ColumnMapping.from_json(json.loads(p.read_text(encoding="utf-8"))))
    return out


def detect_mapping(columns: list[str], mappings: list[ColumnMapping] | None = None) -> ColumnMapping | None:
    """Pick the built-in mapping whose detect_columns are all present."""
    cols = set(columns)
    for m in mappings if mappings is not None else load_builtin_mappings():
        if m.detect_columns and all(c in cols for c in m.detect_columns):
            return m
    return None


# Alias table for user-supplied files. Matching is case-insensitive on a
# normalized name (non-alphanumerics -> '_'). A unit hint in the header such
# as "Speed (mph)" or "Coolant [F]" selects the source unit.
_ALIASES: dict[str, list[str]] = {
    "rpm": ["rpm", "engine_rpm", "engine_speed", "enginerpm"],
    "speed_kmh": ["speed", "speed_kmh", "vehicle_speed", "vss", "speed_km_h", "speed_mph"],
    "coolant_temp_c": ["coolant", "coolant_temp", "coolant_temp_c", "coolant_temperature", "ect",
                       "engine_coolant_temperature"],
    "engine_load_pct": ["engine_load", "engine_load_pct", "load", "load_pct", "calculated_engine_load"],
    "intake_air_temp_c": ["intake_air_temp", "intake_air_temp_c", "iat", "intake_air_temperature"],
    "ambient_air_temp_c": ["ambient_air_temp", "aat", "ambient_temperature", "ambient_air_temperature"],
    "battery_voltage": ["battery_voltage", "control_module_voltage", "vpwr", "voltage", "battery"],
    "throttle_pct": ["throttle", "throttle_pct", "throttle_position", "absolute_throttle_position", "tps"],
    "manifold_pressure_kpa": ["map", "manifold_pressure", "intake_manifold_pressure",
                              "intake_manifold_absolute_pressure", "manifold_pressure_kpa"],
    "baro_pressure_kpa": ["baro", "barometric_pressure", "absolute_barometric_pressure"],
    "mass_air_flow_g_per_s": ["maf", "mass_air_flow", "air_flow_rate", "mass_air_flow_g_per_s"],
    "fuel_level_pct": ["fuel_level", "fuel_level_pct", "fuel_tank", "fuel_tank_level"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "dtc_codes": ["dtc", "dtc_codes", "dtcs", "trouble_codes"],
}
_TIME_ALIASES_ABS = ["timestamp", "datetime", "date_time", "time_stamp", "gps_time", "device_time"]
_TIME_ALIASES_REL = ["t_s", "time_s", "elapsed_s", "seconds", "time", "elapsed", "run_time"]
_UNIT_HINTS = {"mph": "mph", "km_h": "km/h", "kmh": "km/h", "kph": "km/h", "f": "degF",
               "degf": "degF", "c": "degC", "degc": "degC", "psi": "psi", "kpa": "kPa",
               "bar": "bar", "mpa": "MPa", "v": "V", "mv": "mV", "g_s": "g/s", "g_sec": "g/s", "kg_h": "kg/h",
               "inhg": "inHg", "lb_min": "lb/min"}


def _norm(name: str) -> tuple[str, str | None]:
    """Normalize a header and extract a unit hint from (...) or [...] suffix."""
    unit = None
    m = re.search(r"[\(\[]\s*([^\)\]]+?)\s*[\)\]]\s*$", name)
    base = name
    if m:
        # "℉"/"℃" are single Unicode characters (U+2109/U+2103), not "°F" - normalise them first,
        # otherwise a Fahrenheit column would silently be read as Celsius.
        raw_unit = m.group(1).replace("℉", "degF").replace("℃", "degC").replace("°", "deg")
        key = re.sub(r"[^a-z0-9]+", "_", raw_unit.lower()).strip("_")
        unit = _UNIT_HINTS.get(key)
        base = name[:m.start()]
    base = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    if unit is None:
        for suffix, u in (("_mph", "mph"), ("_kmh", "km/h"), ("_f", "degF"), ("_psi", "psi")):
            if base.endswith(suffix):
                unit = u
    return base, unit


def auto_mapping(columns: list[str]) -> ColumnMapping:
    """Best-effort alias mapping for an unknown file. The user can edit it in the UI."""
    rules: dict[str, ColumnRule] = {}
    taken: set[str] = set()
    time_col, basis = None, TimeBasis.SYNTHETIC_CLOCK
    for col in columns:
        base, unit_hint = _norm(col)
        if time_col is None and base in _TIME_ALIASES_ABS:
            time_col, basis = col, TimeBasis.ABSOLUTE
            continue
        if time_col is None and base in _TIME_ALIASES_REL:
            time_col, basis = col, TimeBasis.RELATIVE_RECORDED
            continue
        for fname, aliases in _ALIASES.items():
            if fname in taken:
                continue
            if base in aliases:
                spec = CANONICAL_FIELDS[fname]
                unit = unit_hint or ("mph" if base.endswith("mph") else spec.unit)
                if spec.numeric and unit not in UNITS_BY_FIELD.get(fname, []) \
                        and _IDENTITY.get(unit, unit) != spec.unit:
                    unit = spec.unit  # hint not applicable to this field
                rules[col] = ColumnRule(fname, unit, unit_inferred=unit_hint is None)
                taken.add(fname)
                break
    return ColumnMapping(
        mapping_id="auto_alias",
        dataset_id=None,
        description="Automatic alias-based mapping. Units without an explicit header hint are "
                    "ASSUMED to be canonical units - review them before analysis.",
        detect_columns=[],
        time_column=time_col,
        time_basis=basis,
        columns=rules,
    )


def parse_time_column(raw: pd.Series, basis: TimeBasis, unit: str = "s") -> tuple[np.ndarray, str | None, int]:
    """Return (t_s float array, ISO start timestamp or None, n_unparseable)."""
    if basis == TimeBasis.ABSOLUTE:
        ts = pd.to_datetime(raw, errors="coerce", utc=False, format="mixed")
        bad = int(ts.isna().sum() - raw.isna().sum())
        first = ts.dropna().min() if ts.notna().any() else None
        t = (ts - first).dt.total_seconds().to_numpy(dtype=float) if first is not None \
            else np.full(len(raw), np.nan)
        return t, (first.isoformat() if first is not None else None), max(bad, 0)
    if unit == "time_of_day":
        # "HH:MM:SS(.fff)" with NO date: seconds since the first row, unwrapping a midnight
        # crossing. A calendar date is never invented (time basis stays relative).
        parts = raw.astype("string").str.extract(r"^\s*(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\s*$")
        sec = parts[0].astype(float) * 3600 + parts[1].astype(float) * 60 + parts[2].astype(float)
        v = sec.to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.any():
            first = v[ok][0]
            wrap = np.cumsum(np.r_[0, np.diff(np.where(ok, v, np.nan)) < -43200].astype(float)) * 86400
            v = v + np.nan_to_num(wrap) - first
        bad = int((~ok).sum() - raw.isna().sum())
        return v, None, max(bad, 0)
    num = pd.to_numeric(raw, errors="coerce")
    bad = int(num.isna().sum() - raw.isna().sum())
    scale = {"s": 1.0, "ms": 1e-3, "min": 60.0}.get(unit, 1.0)
    return num.to_numpy(dtype=float) * scale, None, max(bad, 0)
