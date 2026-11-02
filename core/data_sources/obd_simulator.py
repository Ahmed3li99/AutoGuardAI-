"""A SIMULATED ELM327 adapter for demonstrations and tests.

It mimics the parts of the python-OBD API that OBDDataSource uses and answers
queries from a REAL recorded carOBD trip, one row per second of wall time
(optionally faster). Sessions recorded through it are labelled SYNTHETIC with
a 'simulated adapter' note - they are never presented as live car data."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pandas as pd

from core.data_sources.obd_optional import READ_ONLY_PIDS
from core.paths import SAMPLE_DIR

SIM_TRIP = SAMPLE_DIR / "toyota_carobd_long10.csv"


class _Q:
    def __init__(self, v: float, unit: str):
        self.v, self.unit = v, unit

    def to(self, unit: str):
        return SimpleNamespace(magnitude=self.v)   # trip values are already in canonical units


class _Resp:
    def __init__(self, value):
        self.value = value

    def is_null(self):
        return self.value is None


class _Conn:
    def __init__(self, df: pd.DataFrame, speedup: float, clock):
        self.df, self.speedup, self.clock = df, speedup, clock
        self.t0 = clock()
        self.open = True

    def is_connected(self):
        return self.open

    def status(self):
        return "Car Connected (SIMULATED)"

    def port_name(self):
        return "SIMULATED"

    def protocol_name(self):
        return "Simulated ISO 15765-4 (CAN 11/500)"

    def supports(self, cmd):
        if cmd.name == "VIN":
            return False
        field = READ_ONLY_PIDS.get(cmd.name, (None,))[0]
        return cmd.name == "GET_DTC" or (field in self.df.columns and self.df[field].notna().any())

    def _row(self):
        i = int((self.clock() - self.t0) * self.speedup) % len(self.df)
        return self.df.iloc[i]

    def query(self, cmd):
        if cmd.name == "GET_DTC":
            return _Resp([])                      # the recorded trip carries no trouble codes
        field = READ_ONLY_PIDS[cmd.name][0]
        v = self._row().get(field)
        return _Resp(None if v is None or v != v else _Q(float(v), READ_ONLY_PIDS[cmd.name][1]))

    def close(self):
        self.open = False


def simulated_obd_module(speedup: float = 1.0, clock=time.time) -> SimpleNamespace:
    from core.data_sources.files import RecordedFileSource
    from core.preprocessing.resample import resample_1hz
    s = RecordedFileSource(SIM_TRIP).read_session()
    df, _ = resample_1hz(s.data, s.signals)
    names = list(READ_ONLY_PIDS) + ["GET_DTC", "VIN"]
    commands = SimpleNamespace(**{n: SimpleNamespace(name=n) for n in names})
    return SimpleNamespace(OBD=lambda port=None, fast=False: _Conn(df, speedup, clock), commands=commands,
                           scan_serial=lambda: ["SIMULATED"])
