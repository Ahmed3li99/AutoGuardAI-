"""'Learn your car': reference trips, safeguards, personal model training and use."""
import pytest

from ai.model_registry import ModelRegistry
from ai.personal import has_personal_model, readiness, split_reference, train_personal_model
from ai.pipeline import run_analysis
from core.data_sources.files import RecordedFileSource
from core.data_sources.synthetic_demo import SyntheticDemoDataSource
from database.connection import connect
from database.repositories import Repository
from tests.conftest import ROOT

TRIPS = ROOT / "data" / "raw" / "eron93br_carOBD" / "obdiidata"
needs_raw = pytest.mark.skipif(not TRIPS.exists(), reason="raw carOBD trips not downloaded")


def _add(repo, path_or_source, vid):
    src = path_or_source if hasattr(path_or_source, "read_session") else RecordedFileSource(path_or_source)
    s = src.read_session()
    sid = repo.save_session(s, src.quality.to_dict() if src.quality else None, vid)
    repo.save_analysis(sid, run_analysis(s, ModelRegistry(enable_moment=False)))
    return sid, s


def test_split_never_reuses_a_trip():
    tr, cal = split_reference([5, 1, 3, 9])
    assert tr == [1, 3] and cal == [5, 9] and not set(tr) & set(cal)
    assert split_reference([1, 2, 3]) == ([1, 2], [3])


def test_trip_with_warning_cannot_be_reference(tmp_path):
    repo = Repository(connect(tmp_path / "p.sqlite3"))
    vid = repo.save_vehicle("Test car")
    sid, _ = _add(repo, SyntheticDemoDataSource(), vid)          # demo has a coolant + voltage warning
    with pytest.raises(ValueError, match="confirmed warning"):
        repo.set_reference(sid, True)
    unassigned = repo.save_session(SyntheticDemoDataSource("generated_normal_drive", 300).read_session())
    with pytest.raises(ValueError, match="Analyse the trip"):
        repo.set_reference(unassigned, True)


@needs_raw
def test_learn_your_car_end_to_end(tmp_path):
    repo = Repository(connect(tmp_path / "p.sqlite3"))
    vid = repo.save_vehicle("My Etios", "Toyota", "Etios", 2014)
    refs = [_add(repo, TRIPS / n, vid)[0] for n in ("live14.csv", "live18.csv", "long4.csv")]
    assert not readiness(repo, vid)["ready"]
    for sid in refs:
        repo.set_reference(sid, True)
    r = readiness(repo, vid)
    assert r["ready"] and r["trips"] == 3 and r["minutes"] >= 30
    info = train_personal_model(repo, vid)
    assert has_personal_model(vid)
    assert info["train_trips"] == 2 and info["calibration_trips"] == 1
    # adapted (not refitted), evidence-gated: the SAME car as the base model shows no clear
    # systematic difference, so (almost) nothing is adapted
    assert all(0.5 <= b <= 2.0 for _, b in info["car_map"].values())
    assert len(info["car_map"]) <= 2
    assert info["personal_alerts_per_hour_on_calibration"] <= info["base_alerts_per_hour_on_calibration"] + 1
    assert repo.vehicle_model(vid)["version"].startswith("personal-vehicle")

    # a new trip of the same car is analysed with the PERSONAL model
    sid, s = _add(repo, TRIPS / "live19.csv", vid)
    res = run_analysis(s, ModelRegistry(enable_moment=False, vehicle_id=vid))
    st = res.detector_status["expected_vs_actual"]
    assert st["scope"] == "vehicle" and st["calibration"] == "calibrated_personal"
    # ... and a trip of ANOTHER car still uses the base model
    other = run_analysis(s, ModelRegistry(enable_moment=False, vehicle_id=None))
    assert other.detector_status["expected_vs_actual"]["scope"] == "base"


@needs_raw
def test_adaptation_learns_a_different_car():
    """Simulated other car (RPM x1.18, coolant +6 C, voltage -0.35 V): the learned map
    recovers the change and the false-alert rate drops versus the base model."""
    import numpy as np
    from ai.normal_behavior import NormalBehaviorModel
    from ai.personal import adapt_base_model
    from core.paths import MODELS_DIR
    from core.preprocessing.resample import resample_1hz

    def car(name):
        s = RecordedFileSource(TRIPS / name).read_session()
        d, _ = resample_1hz(s.data, s.signals)
        d["rpm"] *= 1.18
        d["coolant_temp_c"] += 6.0
        d["battery_voltage"] -= 0.35
        return d
    base = NormalBehaviorModel.load(MODELS_DIR / "normal_behavior")
    m = adapt_base_model(base, [car("live14.csv"), car("long4.csv")], [car("drive12.csv")], {"version": "t"})
    a, b = m.meta["car_map"]["rpm"]
    assert 1.05 < b < 1.3
    ac, bc = m.meta["car_map"]["coolant_temp_c"]
    assert 3 < (ac + bc * 88) - 88 < 9                        # ~ +6 C at a warm engine
    held = [car(n) for n in ("live19.csv", "long10.csv", "ufpe8.csv")]
    assert sum(len(m.alerts(d)) for d in held) < sum(len(base.alerts(d)) for d in held)
    assert np.isfinite(m.models["rpm"].scale)


@needs_raw
def test_idle_only_learning_trips_are_refused(tmp_path):
    """Measured failure mode: learning only from parked trips -> 20 false alerts/h. The app refuses it."""
    repo = Repository(connect(tmp_path / "i.sqlite3"))
    vid = repo.save_vehicle("Idle car")
    refs = [_add(repo, TRIPS / n, vid)[0] for n in ("idle17.csv", "idle34.csv", "live29.csv")]
    for sid in refs:
        repo.set_reference(sid, True)
    r = readiness(repo, vid)
    assert r["minutes"] >= 30 and r["driving_ok"] is False and not r["ready"]
    with pytest.raises(ValueError, match="real driving"):
        train_personal_model(repo, vid)
