"""Other real cars: Ford / Chrysler import, model choice by available sensors, and how findings of a
model calibrated on a different car are presented."""
import numpy as np
import pandas as pd
import pytest

from ai.explain import findings, verdict
from ai.model_registry import ModelRegistry, choose_variant
from ai.pipeline import run_analysis
from core.data_sources.files import RecordedFileSource
from core.paths import MODELS_DIR
from core.schema import DataOrigin, NormalizedSession, SessionMetadata, TimeBasis

needs_models = pytest.mark.skipif(not (MODELS_DIR / "normal_behavior_drivetrain" / "nbm.joblib").exists()
                                  or not (MODELS_DIR / "normal_behavior" / "nbm.joblib").exists(),
                                  reason="models not trained")


def test_ford_car_scanner_file_imports_with_fahrenheit_mph_and_clock_time(tmp_path):
    # Car Scanner layout: one PID per row, clock time without a date, crossing midnight
    p = tmp_path / "ford.txt"
    p.write_text('time,"Engine RPM (rpm)","Vehicle speed (mph)","Engine coolant temperature (℉)",'
                 '"Absolute load value (%)"\n'
                 '23:59:58.500,800,,,\n23:59:58.700,,10,,\n23:59:59.100,,,194,\n'
                 '00:00:00.200,820,,,55\n00:00:01.000,,20,,\n', encoding="utf-8")
    s = RecordedFileSource(p).read_session()
    assert s.meta.mapping_id == "apmonitor_ford_carscanner_v1"
    assert s.meta.time_basis == TimeBasis.RELATIVE_RECORDED
    t = s.data["t_s"].to_numpy()
    assert t[0] == 0 and np.all(np.diff(t) >= 0) and t[-1] == pytest.approx(2.5)   # midnight unwrapped
    assert s.data["speed_kmh"].dropna().iloc[0] == pytest.approx(16.09, abs=0.01)
    assert s.data["coolant_temp_c"].dropna().iloc[0] == pytest.approx(90.0)          # 194 °F, not 194 °C
    assert "engine_load_pct" not in s.signals            # absolute load is not calculated load


def test_chrysler_file_imports_inhg_and_leading_spaces(tmp_path):
    p = tmp_path / "chrysler.txt"
    p.write_text("﻿Time (sec), Vehicle speed (MPH), Engine RPM (RPM), Engine coolant temperature (°F),"
                 " Intake manifold absolute pressure (inHg)\n0,0,700,190,10\n1,10,1500,190,20\n", encoding="utf-8")
    s = RecordedFileSource(p).read_session()
    assert s.meta.mapping_id == "apmonitor_chrysler_v1"
    assert s.data["manifold_pressure_kpa"].iloc[1] == pytest.approx(67.73, abs=0.01)
    assert s.data["coolant_temp_c"].iloc[0] == pytest.approx(87.78, abs=0.01)


def test_model_variant_follows_available_sensors():
    full = ["rpm", "speed_kmh", "coolant_temp_c", "engine_load_pct", "throttle_pct", "intake_air_temp_c"]
    assert choose_variant(full) == "full"                                  # 6 of 8 inputs
    assert choose_variant(["rpm", "speed_kmh", "throttle_pct", "mass_air_flow_g_per_s"]) == "drivetrain"
    assert choose_variant(["rpm", "speed_kmh"]) == "drivetrain"            # e.g. the VED fleet
    assert choose_variant(["coolant_temp_c", "battery_voltage"]) == "full"  # no RPM+speed: nothing to gain
    assert choose_variant(None) == "full"


def _rpm_speed_trip(n=900, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    speed = np.clip(50 + 30 * np.sin(t / 90) + rng.normal(0, 1, n), 0, None)
    rpm = 800 + speed * 28 + rng.normal(0, 30, n)
    return pd.DataFrame({"t_s": t, "rpm": rpm, "speed_kmh": speed})


@needs_models
def test_rpm_speed_only_log_uses_drivetrain_model_and_marks_other_car():
    df = _rpm_speed_trip()
    df.loc[400:460, "rpm"] = 6500                 # far from anything plausible at that speed
    sess = NormalizedSession(df, SessionMetadata(source_name="ved.csv", dataset_id="ved",
                                                 origin=DataOrigin.RECORDED, time_basis=TimeBasis.RELATIVE_RECORDED))
    res = run_analysis(sess, ModelRegistry(moment_dir=MODELS_DIR / "nonexistent"))
    st = res.detector_status["expected_vs_actual"]
    assert st["status"] == "ok" and st["variant"] == "drivetrain"
    assert st["calibration"] == "uncalibrated_for_this_data"
    fs = findings(res, sess)
    ml = [f for f in fs if f.detectors == ["model:expected_vs_actual"]]
    assert ml and all(f.uncalibrated_only and f.confidence == "low" for f in ml)
    v = verdict(fs, res, sess)
    assert "learn your car" in v["en_sub"].lower()


def test_hybrid_profile_does_not_flag_engine_off_while_moving():
    from ai.consistency import ConsistencyConfig, run_consistency
    df = _rpm_speed_trip(300)
    df.loc[100:160, "rpm"] = 0.0                  # engine off at ~50 km/h: normal for a hybrid
    assert any(a.detector == "consistency:moving_rpm0" for a in run_consistency(df))
    assert not run_consistency(df, ConsistencyConfig(engine_can_stop_while_moving=True))


def test_vehicle_powertrain_reaches_the_analysis(tmp_path):
    from database.connection import connect
    from database.repositories import Repository
    repo = Repository(connect(tmp_path / "t.sqlite3"))
    vid = repo.save_vehicle("Prius", fuel="hybrid")
    assert repo.vehicle_fuel(vid) == "hybrid"
    repo.set_vehicle_fuel(vid, "ice")
    assert repo.vehicle_fuel(vid) == "ice" and repo.vehicle_fuel(None) == ""
    df = _rpm_speed_trip(300)
    df.loc[100:160, "rpm"] = 0.0
    sess = NormalizedSession(df, SessionMetadata(source_name="x.csv", dataset_id="ved", origin=DataOrigin.RECORDED,
                                                 time_basis=TimeBasis.RELATIVE_RECORDED))
    reg = ModelRegistry(moment_dir=MODELS_DIR / "nonexistent", powertrain="hybrid")
    res = run_analysis(sess, reg)
    assert not any(a.detector == "consistency:moving_rpm0" for a in res.alerts)
    assert "Hybrid" in res.detector_status["consistency"]["note"]


@needs_models
def test_learning_a_noisier_car_widens_margins_within_limits():
    from ai.model_registry import NBM_DRIVETRAIN_DIR
    from ai.normal_behavior import NormalBehaviorModel
    from ai.personal import NOISE_UP_TO, adapt_base_model
    base = NormalBehaviorModel.load(NBM_DRIVETRAIN_DIR)
    trips = [_rpm_speed_trip(900, seed=i) for i in range(4)]
    for f in trips:                        # a much less predictable car (other gearbox)
        f["rpm"] += np.random.default_rng(1).normal(0, 600, len(f))
    m = adapt_base_model(base, trips[:3], trips[3:], {"version": "t", "scope": "vehicle"})
    for t, mm in m.models.items():
        s0 = base.models[t].scale
        assert s0 * 0.8 - 1e-9 <= mm.scale <= NOISE_UP_TO * s0 + 1e-9
    assert m.models["rpm"].scale > base.models["rpm"].scale
    off = adapt_base_model(base, trips[:3], trips[3:], {"version": "t", "scope": "vehicle"}, noise_up_to=0.0)
    assert off.models["rpm"].scale <= 1.2 * base.models["rpm"].scale + 1e-9   # the first design
