"""Replay engine controls, resampling, splits, short windows."""
import numpy as np
import pandas as pd
import pytest

from core.playback.replay import ReplayEngine, ReplayState
from core.preprocessing.features import acceleration_mps2
from core.preprocessing.resample import resample_1hz
from core.preprocessing.splits import make_trip_split
from core.schema import DataOrigin, NormalizedSession, SessionMetadata, TimeBasis


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def session(n=100, dt=1.0):
    df = pd.DataFrame({"t_s": np.arange(n) * dt, "rpm": np.arange(n, dtype=float)})
    return NormalizedSession(df, SessionMetadata("x", DataOrigin.RECORDED, TimeBasis.RELATIVE_RECORDED))


def test_play_pause_resume_seek_stop():
    clk = FakeClock()
    e = ReplayEngine(session(), clock=clk)
    e.play()
    clk.t = 10.0
    rows = e.tick()
    assert len(rows) == 11 and e.index == 11            # t=0..10 inclusive
    e.pause()
    clk.t = 50.0
    assert len(e.tick()) == 0 and e.state == ReplayState.PAUSED
    e.resume()
    clk.t = 52.0
    assert e.tick()["t_s"].tolist() == [11.0, 12.0]
    e.set_speed(4.0)
    clk.t = 53.0
    assert e.tick()["t_s"].tolist() == [13.0, 14.0, 15.0, 16.0]
    e.seek(80.0)
    assert e.index == 81 and e.current_row()["t_s"] == 80.0
    e.stop()
    assert e.index == 0 and e.state == ReplayState.STOPPED


def test_finish_and_restart_repeatable():
    clk = FakeClock()
    e = ReplayEngine(session(20), clock=clk)
    e.play()
    clk.t = 100
    first = e.tick()
    assert e.state == ReplayState.FINISHED and len(first) == 20
    e.play()                                             # restarts from the beginning
    clk.t = 200
    second = e.tick()
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_uses_recorded_irregular_timing():
    df = pd.DataFrame({"t_s": [0, 0.5, 3.0, 3.2], "rpm": [1, 2, 3, 4.0]})
    e = ReplayEngine(NormalizedSession(df, SessionMetadata("x", DataOrigin.RECORDED, TimeBasis.RELATIVE_RECORDED)),
                     clock=(c := FakeClock()))
    e.play()
    c.t = 1.0
    assert len(e.tick()) == 2
    c.t = 2.9
    assert len(e.tick()) == 0


def test_invalid_speed_and_empty_session():
    e = ReplayEngine(session(5), clock=FakeClock())
    with pytest.raises(ValueError):
        e.set_speed(3.0)
    with pytest.raises(ValueError):
        ReplayEngine(session(0))


def test_resample_last_per_second_and_gap_policy():
    df = pd.DataFrame({"t_s": [0, 0, 1, 1, 2, 10], "rpm": [1, 2, 3, 4, 5, 6.0]})
    r, info = resample_1hz(df, ["rpm"], ffill_limit_s=3)
    assert r["rpm"].iloc[:3].tolist() == [2, 4, 5]
    assert r["rpm"].iloc[3:6].tolist() == [5, 5, 5]      # short gap filled (limit 3)
    assert r["rpm"].iloc[6:10].isna().all()              # long gap stays missing
    assert info.filled_cells == 3


def test_acceleration_uses_real_time():
    df = pd.DataFrame({"t_s": [0, 2, 4], "speed_kmh": [0, 36, 36]})
    a = acceleration_mps2(df)
    assert np.isnan(a.iloc[0]) and a.iloc[1] == pytest.approx(5.0) and a.iloc[2] == 0


def test_trip_split_disjoint_deterministic_and_stratified():
    names = [f"{c}{i}.csv" for c in ("idle", "drive", "live") for i in range(1, 11)]
    s1, s2 = make_trip_split(names), make_trip_split(names)
    assert s1 == s2
    parts = [set(v) for v in s1.values()]
    assert sum(len(p) for p in parts) == len(names)
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    for cat in ("idle", "drive", "live"):
        assert any(n.startswith(cat) for n in s1["test"])
