"""Clearly labelled SYNTHETIC demo data, deterministic, so the replay/alert
UI can be demonstrated without any vehicle, adapter or file.

Two kinds of scenario:
  real_trip_*  - a REAL recorded carOBD trip (bundled sample long10) with
                 documented faults injected by this program. Realistic driving,
                 artificial faults. Still labelled SYNTHETIC everywhere.
  generated_*  - fully generated driving (fallback when the sample is absent).
Neither is a measurement of a real fault."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.analytics.injection import InjectedEvent, inject
from core.data_sources.base import SourceCapabilities, VehicleDataSource
from core.paths import SAMPLE_DIR
from core.schema import TIME_COL, DataOrigin, NormalizedSession, SessionMetadata, TimeBasis
from core.validation.quality import QualityReport, build_quality_report

BASE_TRIP = SAMPLE_DIR / "toyota_carobd_long10.csv"

REAL_TRIP_FAULTS = [
    InjectedEvent("flatline", "rpm", 700, 760, "Injected: RPM reading frozen while the car keeps moving"),
    InjectedEvent("coolant_overheat", "coolant_temp_c", 900, 1140,
                  "Injected: coolant rises to 112 °C and holds"),
    InjectedEvent("voltage_drop", "battery_voltage", 1200, 1300,
                  "Injected: charging voltage drops to ~11.4 V with engine running"),
]

SCENARIOS = {
    "real_trip_faults": REAL_TRIP_FAULTS,
    "real_trip_normal": [],
    "generated_normal_drive": [],
    "generated_overheat_and_low_voltage": [
        InjectedEvent("coolant_overheat", "coolant_temp_c", 780, 1020,
                      "Injected: coolant rises to 112 °C and holds"),
        InjectedEvent("voltage_drop", "battery_voltage", 1080, 1180,
                      "Injected: charging voltage drops to ~11.4 V with engine running"),
        InjectedEvent("flatline", "rpm", 500, 560,
                      "Injected: RPM reading frozen (stuck-sensor pattern)"),
    ],
}


def generate_base(duration_s: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(duration_s, dtype=float)
    # Speed: idle warm-up, then urban stop/go segments and a faster stretch.
    speed = np.zeros(duration_s)
    seg_t = 120
    while seg_t < duration_s - 60:
        length = int(rng.integers(60, 180))
        peak = float(rng.choice([30, 45, 60, 85]))
        prof = peak * np.sin(np.linspace(0, np.pi, length)) ** 0.6
        end = min(seg_t + length, duration_s)
        speed[seg_t:end] = prof[: end - seg_t]
        seg_t = end + int(rng.integers(10, 40))
    speed = np.clip(speed + rng.normal(0, 0.6, duration_s), 0, None)
    gear_ratio = np.select([speed < 15, speed < 30, speed < 50, speed < 70], [95, 55, 38, 30], 26)
    rpm = np.where(speed < 1, 780, 780 + speed * gear_ratio * 0.9) + rng.normal(0, 15, duration_s)
    accel = np.gradient(speed / 3.6)
    load = np.clip(22 + 0.35 * speed + 55 * np.clip(accel, 0, None) + rng.normal(0, 2, duration_s), 8, 100)
    throttle = np.clip(15.7 + 0.25 * load + rng.normal(0, 0.5, duration_s), 13, 100)
    coolant = 88 - (88 - 32) * np.exp(-t / 260.0) + rng.normal(0, 0.3, duration_s)
    iat = 30 + 8 * (1 - np.exp(-t / 400)) + rng.normal(0, 0.4, duration_s)
    volt = 13.8 + rng.normal(0, 0.05, duration_s)
    volt[:3] = 12.4  # before alternator output stabilises
    mapk = np.clip(25 + 0.7 * load + rng.normal(0, 1, duration_s), 15, 102)
    return pd.DataFrame({TIME_COL: t, "rpm": np.round(rpm, 2), "speed_kmh": np.round(speed),
                         "coolant_temp_c": np.round(coolant, 1), "engine_load_pct": np.round(load, 2),
                         "throttle_pct": np.round(throttle, 2), "intake_air_temp_c": np.round(iat, 1),
                         "battery_voltage": np.round(volt, 3), "manifold_pressure_kpa": np.round(mapk)})


class SyntheticDemoDataSource(VehicleDataSource):
    def __init__(self, scenario: str = "real_trip_faults", duration_s: int = 1500, seed: int = 42):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario}; choose from {list(SCENARIOS)}")
        if scenario.startswith("real_trip") and not BASE_TRIP.exists():
            scenario = "generated_" + ("overheat_and_low_voltage" if scenario.endswith("faults") else "normal_drive")
        self.scenario, self.duration_s, self.seed = scenario, duration_s, seed
        self.name = f"synthetic:{scenario}"
        self.events = SCENARIOS[scenario]
        self.quality: QualityReport | None = None

    def _base(self) -> tuple[pd.DataFrame, str | None, list[str]]:
        if self.scenario.startswith("real_trip"):
            from core.data_sources.files import RecordedFileSource
            from core.preprocessing.resample import resample_1hz
            rec = RecordedFileSource(BASE_TRIP).read_session()
            df, _ = resample_1hz(rec.data, rec.signals)
            return df, "toyota_carobd", [
                f"SYNTHETIC DEMO: real recorded trip {BASE_TRIP.name} (carOBD, Toyota Etios) on a 1 Hz grid, "
                "with faults INJECTED by AutoGuard AI. The faults are artificial - not a real vehicle failure."]
        return generate_base(self.duration_s, self.seed), "synthetic_demo", [
            f"SYNTHETIC DEMO DATA generated by AutoGuard AI (seed {self.seed}) - not a real vehicle."]

    def read_session(self) -> NormalizedSession:
        base, dataset_id, notes = self._base()
        df = inject(base, self.events)
        meta = SessionMetadata(
            source_name=self.name, origin=DataOrigin.SYNTHETIC, time_basis=TimeBasis.SYNTHETIC_CLOCK,
            dataset_id=dataset_id, mapping_id="canonical", playback_interval_s=1.0,
            notes=notes + [f"{e.description} at {e.start_s:.0f}-{e.end_s:.0f} s" for e in self.events],
            injected_events=[e.to_dict() for e in self.events])
        s = NormalizedSession(df, meta)
        self.quality = build_quality_report(s, len(df), {}, {}, 0, [])
        return s

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(DataOrigin.SYNTHETIC, TimeBasis.SYNTHETIC_CLOCK,
                                  tuple(c for c in generate_base(10).columns if c != TIME_COL))
