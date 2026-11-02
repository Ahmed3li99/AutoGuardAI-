"""Canonical vehicle telemetry schema.

Every data source (recorded file, synthetic demo, future OBD adapter) is
normalized into a pandas DataFrame whose columns are a subset of
``CANONICAL_FIELDS`` plus the time column ``t_s``. Missing signals are
*absent columns* (or NaN), never zeros.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import pandas as pd


class DataOrigin(str, Enum):
    RECORDED = "recorded"          # real measurement from a recorded file
    SYNTHETIC = "synthetic"        # generated demo data - never a real car
    LIVE_OBD = "live_obd"          # future: read from a physical adapter

    @property
    def label(self) -> str:
        return {
            DataOrigin.RECORDED: "Recorded Replay",
            DataOrigin.SYNTHETIC: "Synthetic Demo",
            DataOrigin.LIVE_OBD: "Live OBD-II",
        }[self]


class TimeBasis(str, Enum):
    ABSOLUTE = "absolute"                    # true wall-clock timestamps in the source
    RELATIVE_RECORDED = "relative_recorded"  # recorded elapsed time (e.g. engine run time)
    SYNTHETIC_CLOCK = "synthetic_clock"      # NO time in source: fixed playback interval

    @property
    def label(self) -> str:
        return {
            TimeBasis.ABSOLUTE: "Recorded timestamps",
            TimeBasis.RELATIVE_RECORDED: "Recorded relative time",
            TimeBasis.SYNTHETIC_CLOCK: "Synthetic playback clock (no time in source)",
        }[self]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    unit: str
    description: str
    # Physically possible range for a standard OBD-II PID / sensor. Values
    # outside are flagged as invalid by validation (not silently clipped).
    valid_min: float | None = None
    valid_max: float | None = None
    numeric: bool = True


# Ranges follow the encodable range of the corresponding SAE J1979 Mode 01 PID
# where one exists (e.g. ECT: A-40 => -40..215 C) so that only impossible
# values are flagged; plausibility warnings live in the rules baseline.
CANONICAL_FIELDS: dict[str, FieldSpec] = {f.name: f for f in [
    FieldSpec("rpm", "rpm", "Engine speed", 0, 16383.75),
    FieldSpec("speed_kmh", "km/h", "Vehicle speed", 0, 255),
    FieldSpec("coolant_temp_c", "°C", "Engine coolant temperature", -40, 215),
    FieldSpec("engine_load_pct", "%", "Calculated engine load", 0, 100),
    FieldSpec("intake_air_temp_c", "°C", "Intake air temperature", -40, 215),
    FieldSpec("ambient_air_temp_c", "°C", "Ambient air temperature", -40, 215),
    FieldSpec("battery_voltage", "V", "Control module voltage (PID 0x42)", 0, 65.535),
    FieldSpec("throttle_pct", "%", "Throttle position", 0, 100),
    FieldSpec("manifold_pressure_kpa", "kPa", "Intake manifold absolute pressure", 0, 255),
    FieldSpec("baro_pressure_kpa", "kPa", "Barometric pressure", 0, 255),
    FieldSpec("mass_air_flow_g_per_s", "g/s", "Mass air flow", 0, 655.35),
    FieldSpec("fuel_level_pct", "%", "Fuel tank level input", 0, 100),
    FieldSpec("fuel_rail_pressure_kpa", "kPa", "Fuel rail pressure", 0, 655350),
    FieldSpec("timing_advance_deg", "°", "Timing advance", -64, 63.5),
    FieldSpec("short_fuel_trim_pct", "%", "Short term fuel trim bank 1", -100, 99.2),
    FieldSpec("long_fuel_trim_pct", "%", "Long term fuel trim bank 1", -100, 99.2),
    FieldSpec("catalyst_temp_c", "°C", "Catalyst temperature B1S1", -40, 6513.5),
    FieldSpec("latitude", "deg", "Latitude", -90, 90),
    FieldSpec("longitude", "deg", "Longitude", -180, 180),
    FieldSpec("dtc_codes", "", "Diagnostic trouble codes (as reported)", numeric=False),
]}

NUMERIC_FIELDS = [k for k, v in CANONICAL_FIELDS.items() if v.numeric]

# Signals shown first on the dashboard (only if present in the data).
PRIMARY_SIGNALS = ["rpm", "speed_kmh", "coolant_temp_c", "engine_load_pct",
                   "throttle_pct", "battery_voltage", "intake_air_temp_c",
                   "manifold_pressure_kpa"]

TIME_COL = "t_s"   # seconds; meaning defined by SessionMetadata.time_basis


@dataclass
class SessionMetadata:
    source_name: str                  # file name or "synthetic:<scenario>"
    origin: DataOrigin
    time_basis: TimeBasis
    dataset_id: str | None = None     # e.g. "toyota_carobd" if recognized
    mapping_id: str | None = None
    source_sha256: str | None = None
    # Provenance identifiers ONLY if genuinely present in the source.
    vehicle_id: str | None = None
    trip_id: str | None = None
    start_timestamp: str | None = None  # ISO8601, only for ABSOLUTE basis
    playback_interval_s: float | None = None  # only for SYNTHETIC_CLOCK
    column_map: dict[str, str] = field(default_factory=dict)   # source -> canonical
    unit_conversions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # Only for synthetic demo data: the faults this program injected (ground truth of the demo).
    injected_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["origin"] = self.origin.value
        d["time_basis"] = self.time_basis.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionMetadata":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}   # tolerate older/newer rows
        d["origin"] = DataOrigin(d["origin"])
        d["time_basis"] = TimeBasis(d["time_basis"])
        return cls(**d)


@dataclass
class NormalizedSession:
    """Normalized telemetry plus provenance. ``data`` is sorted by ``t_s``."""
    data: pd.DataFrame
    meta: SessionMetadata

    @property
    def signals(self) -> list[str]:
        return [c for c in self.data.columns
                if c in CANONICAL_FIELDS and c != "dtc_codes"
                and self.data[c].notna().any()]

    @property
    def duration_s(self) -> float:
        if self.data.empty:
            return 0.0
        return float(self.data[TIME_COL].iloc[-1] - self.data[TIME_COL].iloc[0])
