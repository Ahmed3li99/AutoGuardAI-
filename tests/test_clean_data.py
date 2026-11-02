"""Clean data, valid labels and leakage guards for the multi-car training set."""
import json

import numpy as np
import pandas as pd
import pytest

from ai.consistency import ConsistencyConfig, run_consistency
from core.analytics.injection import InjectedEvent, changes_data, inject
from core.paths import MODELS_DIR, resource_root

SPLIT = resource_root() / "datasets" / "splits" / "multicar_split.json"


def _drive(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    speed = 60 + 25 * np.sin(t / 40) + rng.normal(0, 0.5, n)
    rpm = np.round(900 + speed * 26 + rng.normal(0, 20, n))
    return pd.DataFrame({"t_s": t, "rpm": rpm, "speed_kmh": speed})


def test_frozen_rpm_while_driving_is_flagged_but_normal_driving_is_not():
    df = _drive()
    assert not [a for a in run_consistency(df) if a.detector.endswith("_stuck")]
    df.loc[200:229, "rpm"] = df.loc[200, "rpm"]               # 30 s frozen while speed keeps changing
    hits = [a for a in run_consistency(df) if a.detector == "consistency:rpm_stuck"]
    assert hits and hits[0].t_start_s <= 201 and hits[0].t_end_s >= 228
    assert not [a for a in run_consistency(df, ConsistencyConfig(check_stuck=False)) if a.detector.endswith("_stuck")]


def test_rpm_zero_while_moving_is_left_to_the_engine_off_check():
    df = _drive()
    df.loc[200:229, "rpm"] = 0.0
    det = {a.detector for a in run_consistency(df)}
    assert "consistency:moving_rpm0" in det and "consistency:rpm_stuck" not in det


def test_injection_that_changes_nothing_is_not_a_valid_label():
    df = _drive()
    df.loc[100:200, "rpm"] = 800.0                            # already constant (e.g. steady idle)
    ev = InjectedEvent("flatline", "rpm", 120, 180, "t")
    assert not changes_data(df, inject(df, [ev]), ev)
    ev2 = InjectedEvent("flatline", "rpm", 300, 360, "t")
    assert changes_data(df, inject(df, [ev2]), ev2)


@pytest.mark.skipif(not SPLIT.exists(), reason="multi-car split not built")
def test_multicar_split_has_no_leakage():
    s = json.loads(SPLIT.read_text(encoding="utf-8"))
    tr, cal, te = (set(s["cars"][k]) for k in ("train", "calibration", "test"))
    assert not (tr & cal) and not (tr & te) and not (cal & te)
    assert s["leakage_checks"]["trip_fingerprint_overlap_with_test"] == 0
    assert all(v in tr for v, _ in s["kept_trips"]["train"])
    assert all(v in cal for v, _ in s["kept_trips"]["calibration"])


@pytest.mark.skipif(not (MODELS_DIR / "normal_behavior_drivetrain" / "nbm_meta.json").exists(), reason="no model")
def test_shipped_drivetrain_model_is_the_multicar_one():
    meta = json.loads((MODELS_DIR / "normal_behavior_drivetrain" / "nbm_meta.json").read_text(encoding="utf-8"))
    assert "multicar" in meta["version"] and meta["split_file"].endswith("multicar_split.json")


def test_frozen_rpm_while_parked_is_flagged_after_20_s_only():
    df = _drive()
    df["speed_kmh"] = 0.0
    df["rpm"] = 800 + np.random.default_rng(3).normal(0, 8, len(df)).round()
    assert not [a for a in run_consistency(df) if a.detector == "consistency:rpm_stuck"]
    df.loc[300:314, "rpm"] = 812.0                             # 15 s: below the limit
    assert not [a for a in run_consistency(df) if a.detector == "consistency:rpm_stuck"]
    df.loc[400:429, "rpm"] = 812.0                             # 30 s: frozen
    assert [a for a in run_consistency(df) if a.detector == "consistency:rpm_stuck"]
