"""OPTIONAL, READ-ONLY OBD-II source (python-OBD + ELM327-type adapter).

STATUS: NOT TESTED ON A REAL VEHICLE. Only mocked unit tests exist
(tests/test_obd_optional.py). Disabled unless the feature flag is on
(env AUTOGUARD_ENABLE_OBD=1 or Settings > Hardware). Requires the optional
dependency in requirements-obd-optional.txt.

Safety scope: only Mode 01 "current data" PIDs listed in ``READ_ONLY_PIDS``
are ever queried. The class has no code path for clearing DTCs (Mode 04),
actuator tests, ECU programming or raw CAN transmission.
"""
from __future__ import annotations

import datetime as dt
import os
import time
from types import ModuleType
from typing import Callable

import pandas as pd

from core.data_sources.base import DataSourceError, Reading, SourceCapabilities, VehicleDataSource
from core.schema import TIME_COL, DataOrigin, NormalizedSession, SessionMetadata, TimeBasis

# python-OBD command name -> (canonical field, pint unit string for .to())
READ_ONLY_PIDS: dict[str, tuple[str, str]] = {
    "RPM": ("rpm", "rpm"),
    "SPEED": ("speed_kmh", "kph"),
    "COOLANT_TEMP": ("coolant_temp_c", "degC"),
    "ENGINE_LOAD": ("engine_load_pct", "percent"),
    "THROTTLE_POS": ("throttle_pct", "percent"),
    "INTAKE_TEMP": ("intake_air_temp_c", "degC"),
    "INTAKE_PRESSURE": ("manifold_pressure_kpa", "kilopascal"),
    "MAF": ("mass_air_flow_g_per_s", "gps"),
    "FUEL_LEVEL": ("fuel_level_pct", "percent"),
    "CONTROL_MODULE_VOLTAGE": ("battery_voltage", "volt"),
    "BAROMETRIC_PRESSURE": ("baro_pressure_kpa", "kilopascal"),
    "AMBIANT_AIR_TEMP": ("ambient_air_temp_c", "degC"),  # python-OBD spelling
}

FEATURE_FLAG_ENV = "AUTOGUARD_ENABLE_OBD"


def obd_feature_enabled(settings_flag: bool = False) -> bool:
    return settings_flag or os.environ.get(FEATURE_FLAG_ENV) == "1"


def load_obd_module() -> ModuleType | None:
    try:
        import obd  # type: ignore
        return obd
    except ImportError:
        return None


class OBDDataSource(VehicleDataSource):
    name = "Live OBD-II (experimental)"

    def __init__(self, port: str | None = None, *, enabled: bool | None = None,
                 obd_module: ModuleType | None = None, clock: Callable[[], float] = time.time,
                 max_retries: int = 2):
        self.port = port
        self.enabled = obd_feature_enabled() if enabled is None else enabled
        self._obd = obd_module if obd_module is not None else load_obd_module()
        self._clock = clock
        self._max_retries = max_retries
        self._conn = None
        self._start_wall: float | None = None
        self._commands: dict[str, object] = {}
        self.unsupported: list[str] = []
        self.recorded: list[Reading] = []
        self.errors: list[str] = []

    # ---- availability -------------------------------------------------
    def availability(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "Disabled (feature flag off). Live OBD-II is a future, adapter-required feature."
        if self._obd is None:
            return False, "python-OBD not installed (pip install -r requirements-obd-optional.txt)."
        return True, "Available (NOT TESTED ON A REAL VEHICLE)."

    def list_ports(self) -> list[str]:
        ok, msg = self.availability()
        if not ok:
            raise DataSourceError(msg)
        return list(self._obd.scan_serial())

    # ---- connection ---------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._conn is not None and bool(self._conn.is_connected())

    def open(self) -> None:
        ok, msg = self.availability()
        if not ok:
            raise DataSourceError(msg)
        last = "unknown error"
        for _ in range(self._max_retries + 1):
            try:
                conn = self._obd.OBD(self.port, fast=False)
            except Exception as e:  # serial errors vary by platform
                last = f"{type(e).__name__}: {e}"
                continue
            if conn.is_connected():
                self._conn = conn
                break
            last = f"adapter status: {conn.status()}"
            conn.close()
        if self._conn is None:
            self.errors.append(last)
            raise DataSourceError(f"Could not connect to OBD-II adapter on {self.port or 'auto'}: {last}")
        self._discover()
        self._start_wall = self._clock()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def _discover(self) -> None:
        self._commands.clear()
        self.unsupported = []
        for name in READ_ONLY_PIDS:
            cmd = getattr(self._obd.commands, name, None)
            if cmd is not None and self._conn.supports(cmd):
                self._commands[name] = cmd
            else:
                self.unsupported.append(name)

    def supported_signals(self) -> list[str]:
        return [READ_ONLY_PIDS[n][0] for n in self._commands]

    # ---- polling ------------------------------------------------------
    def poll_once(self) -> Reading:
        if not self.is_connected:
            try:
                self.close()
                self.open()
            except DataSourceError as e:
                raise DataSourceError(f"Connection lost and reconnect failed: {e}") from e
        values: dict[str, float] = {}
        for name, cmd in self._commands.items():
            field, unit = READ_ONLY_PIDS[name]
            try:
                resp = self._conn.query(cmd)
            except Exception as e:
                self.errors.append(f"{name}: {type(e).__name__}: {e}")
                continue
            if resp is None or resp.is_null() or resp.value is None:
                continue
            v = resp.value
            try:
                values[field] = float(v.to(unit).magnitude) if hasattr(v, "to") else float(v)
            except Exception as e:
                self.errors.append(f"{name}: unit conversion failed: {e}")
        reading = Reading(self._clock() - (self._start_wall or self._clock()), values,
                          DataOrigin.LIVE_OBD, unsupported=list(self.unsupported))
        self.recorded.append(reading)
        return reading

    # ---- VehicleDataSource ------------------------------------------
    def read_session(self) -> NormalizedSession:
        rows = [{TIME_COL: r.t_s, **r.values} for r in self.recorded]
        df = pd.DataFrame(rows, columns=[TIME_COL, *[f for f, _ in READ_ONLY_PIDS.values()]]
                          ).dropna(axis=1, how="all")
        if TIME_COL not in df:
            df[TIME_COL] = pd.Series(dtype=float)
        start = (dt.datetime.fromtimestamp(self._start_wall).isoformat()
                 if self._start_wall else None)
        meta = SessionMetadata(
            source_name=f"obd:{self.port or 'auto'}", origin=DataOrigin.LIVE_OBD,
            time_basis=TimeBasis.ABSOLUTE, start_timestamp=start, mapping_id="python-obd",
            notes=["Live OBD-II recording - connector NOT TESTED ON A REAL VEHICLE."]
                  + ([f"Unsupported PIDs: {', '.join(self.unsupported)}"] if self.unsupported else []))
        return NormalizedSession(df, meta)

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(DataOrigin.LIVE_OBD, TimeBasis.ABSOLUTE,
                                  tuple(self.supported_signals()), is_live=True,
                                  supports_seek=False, hardware_tested=False)
