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
                 max_retries: int = 2, simulated: bool = False, active: bool = False):
        self.port = port          # None = let python-OBD auto-detect the adapter's serial port
        self.simulated = simulated
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
        from ai.active_diagnosis import ActiveScheduler
        self.scheduler = ActiveScheduler() if active else None
        self.acquisition_summary: dict = {}

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
        if self._start_wall is None:
            # Start the session clock ONCE. A reconnect keeps it, so time never runs
            # backwards; the outage simply shows up as a gap in the recording.
            self._start_wall = self._clock()
        else:
            self.errors.append(f"Reconnected at t={self._clock() - self._start_wall:.1f} s (gap in recording).")

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

    def identify(self) -> dict:
        """Read-only identification: port, protocol and VIN (Mode 09, if the car reports it)."""
        info = {"port": None, "protocol": None, "vin": None}
        if not self.is_connected:
            return info
        for key, fn in (("port", "port_name"), ("protocol", "protocol_name")):
            try:
                info[key] = getattr(self._conn, fn)()
            except Exception:
                pass
        vin_cmd = getattr(self._obd.commands, "VIN", None)
        if vin_cmd is not None:
            try:
                if self._conn.supports(vin_cmd):
                    r = self._conn.query(vin_cmd)
                    if r is not None and not r.is_null():
                        v = r.value
                        info["vin"] = v.decode(errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
            except Exception as e:
                self.errors.append(f"VIN: {e}")
        return info

    def read_dtcs(self) -> list[tuple[str, str]]:
        """READ the trouble codes the car has stored (Mode 03). Never clears them."""
        cmd = getattr(self._obd.commands, "GET_DTC", None)
        if cmd is None or not self.is_connected:
            return []
        try:
            r = self._conn.query(cmd)
        except Exception as e:
            self.errors.append(f"GET_DTC: {e}")
            return []
        if r is None or r.is_null() or not r.value:
            return []
        return [(str(c), str(d)) for c, d in r.value]

    # ---- polling ------------------------------------------------------
    def poll_once(self) -> Reading:
        if not self.is_connected:
            try:
                self.close()
                self.open()
            except DataSourceError as e:
                raise DataSourceError(f"Connection lost and reconnect failed: {e}") from e
        values: dict[str, float] = {}
        commands = list(self._commands.items())
        if self.scheduler is not None:
            plan = self.scheduler.plan(self.supported_signals(), time.monotonic())
            chosen = {p["signal"] for p in plan}
            order = {p["signal"]: i for i, p in enumerate(plan)}
            commands = sorted([(n,c) for n,c in commands if READ_ONLY_PIDS[n][0] in chosen],
                              key=lambda item: order[READ_ONLY_PIDS[item[0]][0]])
        for name, cmd in commands:
            field, unit = READ_ONLY_PIDS[name]
            started = time.monotonic()
            try:
                resp = self._conn.query(cmd)
            except Exception as e:
                self.errors.append(f"{name}: {type(e).__name__}: {e}")
                if self.scheduler is not None:
                    self.scheduler.observe(field, None, time.monotonic(), time.monotonic()-started)
                continue
            if resp is None or resp.is_null() or resp.value is None:
                if self.scheduler is not None:
                    self.scheduler.observe(field, None, time.monotonic(), time.monotonic()-started)
                continue
            v = resp.value
            try:
                values[field] = float(v.to(unit).magnitude) if hasattr(v, "to") else float(v)
            except Exception as e:
                self.errors.append(f"{name}: unit conversion failed: {e}")
            if self.scheduler is not None:
                self.scheduler.observe(field, values.get(field), time.monotonic(), time.monotonic()-started)
        if self.scheduler is not None:
            self.acquisition_summary = self.scheduler.summary()
        reading = Reading(self._clock() - (self._start_wall if self._start_wall is not None else self._clock()), values,
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
        if self.simulated:
            origin, basis, name = DataOrigin.SYNTHETIC, TimeBasis.SYNTHETIC_CLOCK, "simulated-obd-adapter"
            notes = ["SIMULATED ADAPTER: a recorded carOBD trip replayed through the live OBD code path. "
                     "No real car or adapter was involved."]
            start = None
        else:
            origin, basis, name = DataOrigin.LIVE_OBD, TimeBasis.ABSOLUTE, f"obd:{self.port or 'auto'}"
            notes = ["Live OBD-II recording - connector NOT TESTED ON A REAL VEHICLE."]
        meta = SessionMetadata(
            source_name=name, origin=origin, time_basis=basis, start_timestamp=start, mapping_id="python-obd",
            notes=notes + ([f"Unsupported PIDs: {', '.join(self.unsupported)}"] if self.unsupported else []))
        if self.scheduler is not None:
            meta.notes.append("ACTIVE_SAMPLING: PIDs sampled selectively; absent cells may be unrequested, not sensor failures.")
        return NormalizedSession(df, meta)

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(DataOrigin.LIVE_OBD, TimeBasis.ABSOLUTE,
                                  tuple(self.supported_signals()), is_live=True,
                                  supports_seek=False, hardware_tested=False)
