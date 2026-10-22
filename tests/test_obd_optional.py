"""MOCKED tests of the optional OBD connector. These do not involve any real
adapter or vehicle and are NOT evidence of hardware compatibility."""
from types import SimpleNamespace

import pytest

from core.data_sources.base import DataSourceError
from core.data_sources.obd_optional import READ_ONLY_PIDS, OBDDataSource
from core.schema import DataOrigin


class Q:
    """Minimal stand-in for a pint Quantity."""
    def __init__(self, v, unit):
        self.v, self.unit = v, unit

    def to(self, unit):
        conv = {("mph", "kph"): 1.609344}.get((self.unit, unit), 1.0)
        return SimpleNamespace(magnitude=self.v * conv)


class Resp:
    def __init__(self, value):
        self.value = value

    def is_null(self):
        return self.value is None


class FakeConn:
    def __init__(self, supported, values, connected=True):
        self.supported, self.values, self.connected = supported, values, connected
        self.queries = []
        self.closed = False

    def is_connected(self):
        return self.connected and not self.closed

    def status(self):
        return "Car Connected" if self.connected else "Not Connected"

    def supports(self, cmd):
        return cmd.name in self.supported

    def query(self, cmd):
        self.queries.append(cmd.name)
        if cmd.name == "MAF":
            raise IOError("serial timeout")
        return Resp(self.values.get(cmd.name))

    def close(self):
        self.closed = True


def fake_obd(conns):
    it = iter(conns)
    commands = SimpleNamespace(**{n: SimpleNamespace(name=n) for n in list(READ_ONLY_PIDS) + ["CLEAR_DTC"]})
    return SimpleNamespace(OBD=lambda port, fast=False: next(it), commands=commands,
                           scan_serial=lambda: ["COM3", "COM7"])


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AUTOGUARD_ENABLE_OBD", raising=False)
    src = OBDDataSource(obd_module=fake_obd([]))
    ok, msg = src.availability()
    assert not ok and "Disabled" in msg
    with pytest.raises(DataSourceError):
        src.open()


def test_library_missing_is_reported():
    src = OBDDataSource(enabled=True, obd_module=None)
    src._obd = None
    assert "not installed" in src.availability()[1]


def test_port_selection_connect_discover_poll_disconnect():
    conn = FakeConn({"RPM", "SPEED", "COOLANT_TEMP", "MAF"},
                    {"RPM": Q(850, "rpm"), "SPEED": Q(10, "mph"), "COOLANT_TEMP": Q(88, "degC")})
    clock = iter([100.0, 101.5]).__next__              # open() at 100 s, first poll at 101.5 s
    src = OBDDataSource("COM3", enabled=True, obd_module=fake_obd([conn]), clock=clock)
    assert src.list_ports() == ["COM3", "COM7"]
    src.open()
    assert set(src.supported_signals()) == {"rpm", "speed_kmh", "coolant_temp_c", "mass_air_flow_g_per_s"}
    assert "CONTROL_MODULE_VOLTAGE" in src.unsupported
    r = src.poll_once()
    assert r.origin == DataOrigin.LIVE_OBD and r.t_s == pytest.approx(1.5)
    assert r.values == {"rpm": 850, "speed_kmh": pytest.approx(16.09344), "coolant_temp_c": 88}
    assert any("MAF" in e for e in src.errors)                     # query error recorded, not fatal
    assert "CLEAR_DTC" not in conn.queries                          # read-only
    s = src.read_session()
    assert len(s.data) == 1 and s.meta.origin == DataOrigin.LIVE_OBD
    assert any("NOT TESTED ON A REAL VEHICLE" in n for n in s.meta.notes)
    assert src.capabilities().hardware_tested is False
    src.close()
    assert conn.closed


def test_connect_failure_and_reconnect():
    bad = FakeConn(set(), {}, connected=False)
    src = OBDDataSource("COM9", enabled=True, obd_module=fake_obd([bad, bad, bad]), max_retries=2)
    with pytest.raises(DataSourceError, match="Could not connect"):
        src.open()
    c1 = FakeConn({"RPM"}, {"RPM": Q(900, "rpm")})
    c2 = FakeConn({"RPM"}, {"RPM": Q(910, "rpm")})
    src = OBDDataSource("COM3", enabled=True, obd_module=fake_obd([c1, c2]))
    src.open()
    c1.connected = False                                             # link drops
    assert src.poll_once().values == {"rpm": 910}                   # reconnected transparently


def test_only_mode01_whitelist():
    assert "CLEAR_DTC" not in READ_ONLY_PIDS
    assert all(not n.startswith(("CLEAR", "GET_DTC")) for n in READ_ONLY_PIDS)
