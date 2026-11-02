"""Sensor-consistency checks: "is the car sick, or is the sensor lying?"."""
import numpy as np
import pytest

from ai.consistency import run_consistency
from ai.explain import findings, verdict
from ai.model_registry import ModelRegistry
from ai.pipeline import run_analysis
from core.data_sources.files import RecordedFileSource
from core.paths import SAMPLE_DIR
from core.preprocessing.resample import resample_1hz
from core.schema import NormalizedSession


def _trip(name="toyota_carobd_long10.csv"):
    s = RecordedFileSource(SAMPLE_DIR / name).read_session()
    d, _ = resample_1hz(s.data, s.signals)
    return s, d


@pytest.mark.parametrize("name", ["toyota_carobd_long10.csv", "toyota_carobd_ufpe7.csv", "toyota_carobd_idle3.csv"])
def test_quiet_on_real_trips(name):
    assert run_consistency(_trip(name)[1]) == []


def test_detects_each_contradiction_kind():
    _, d = _trip()
    d = d.copy()
    d.loc[800:820, "coolant_temp_c"] = 160.0          # impossible jump
    d.loc[1500:1530, "rpm"] = 0.0                     # moving at ~80 km/h with RPM 0
    d.loc[2000:2040, "intake_air_temp_c"] = np.nan    # one sensor drops out
    kinds = {a.detector for a in run_consistency(d)}
    assert {"consistency:coolant_temp_c_jump", "consistency:moving_rpm0",
            "consistency:intake_air_temp_c_dropout"} <= kinds


def test_whole_row_gap_is_not_a_sensor_dropout():
    _, d = _trip()
    d = d.copy()
    d.loc[1000:1030, [c for c in d.columns if c != "t_s"]] = np.nan   # logger stopped entirely
    assert not [a for a in run_consistency(d) if a.detector.endswith("_dropout")]


def test_findings_are_categorised_as_data_and_merged():
    s, d = _trip()
    d = d.copy()
    d.loc[1500:1530, "rpm"] = 0.0
    s2 = NormalizedSession(d, s.meta)
    r = run_analysis(s2, ModelRegistry(enable_moment=False))
    fs = [f for f in findings(r, s2) if 1490 <= f.t_start_s <= 1540]
    assert len(fs) == 1                                   # RPM-0 and speed mismatch merged (drivetrain)
    f = fs[0]
    assert f.category == "data" and f.confidence == "high"
    assert "0 RPM" in f.en["title"] and "Meanwhile the other sensors read" in f.en["evidence"]
    assert "measurement or recording problem" in f.en["meaning"]


def test_doubtful_reading_with_rule_warning_stays_serious():
    s, d = _trip()
    d = d.copy()
    d.loc[800:830, "coolant_temp_c"] = 160.0
    s2 = NormalizedSession(d, s.meta)
    r = run_analysis(s2, ModelRegistry(enable_moment=False))
    f = next(f for f in findings(r, s2) if f.signal == "coolant_temp_c")
    assert f.severity == "warning" and f.category == "data"
    assert "doubtful" in f.en["title"] and "dashboard gauge" in f.en["meaning"]
    assert verdict(findings(r, s2), r, s2)["level"] == "warning"


def test_spike_check_flags_glitch_values_inside_the_pid_range():
    from ai.consistency import ConsistencyConfig
    _, d = _trip()
    d = d.copy()
    i = d.index[d["rpm"].notna()][600]
    d.loc[i:i + 1, "rpm"] = 11596.0                  # the kind of glitch seen in a real Ford log
    d.loc[i + 300, "baro_pressure_kpa"] = 210.0
    det = {a.detector for a in run_consistency(d)}
    assert {"consistency:rpm_spike", "consistency:baro_pressure_kpa_spike"} <= det
    assert not any(a.detector.endswith("_spike") for a in run_consistency(d, ConsistencyConfig(check_spikes=False)))


def test_spike_check_quiet_on_a_real_high_rev():
    _, d = _trip()
    d = d.copy()
    i = d.index[d["rpm"].notna()][600]
    d.loc[i:i + 4, "rpm"] = [3500.0, 5000.0, 6500.0, 5200.0, 3600.0]   # a hard rev, within real data
    assert not any(a.detector == "consistency:rpm_spike" for a in run_consistency(d))


def test_spike_finding_is_explained_as_a_reading_problem():
    s, d = _trip()
    data = s.data.copy()
    t0 = float(data["t_s"].iloc[3000])
    data.loc[(data["t_s"] >= t0) & (data["t_s"] < t0 + 2), "rpm"] = 11596.0   # whole seconds (raw is ~2 Hz)
    sess = NormalizedSession(data, s.meta)
    res = run_analysis(sess, ModelRegistry(enable_moment=False))
    f = [x for x in findings(res, sess) if "consistency" in x.detectors and x.signal == "rpm"]
    assert f and f[0].category == "data" and "impossible reading" in f[0].en["title"]
    assert "6,605" in f[0].en["what"]


def test_baro_jump_below_value_limit_is_explained_as_a_jump():
    s, _ = _trip()
    data = s.data.copy()
    t0 = float(data["t_s"].iloc[3000])
    m = (data["t_s"] >= t0) & (data["t_s"] < t0 + 2)
    data.loc[m, "baro_pressure_kpa"] = data.loc[m, "baro_pressure_kpa"] + 8.0   # like 83 -> 91 -> 83 in a real log
    sess = NormalizedSession(data, s.meta)
    res = run_analysis(sess, ModelRegistry(enable_moment=False))
    f = [x for x in findings(res, sess) if x.signal == "baro_pressure_kpa"]
    assert f and "impossible jump" in f[0].en["title"] and "jumped by 8" in f[0].en["what"]
    assert "3 kPa" in f[0].en["what"]
