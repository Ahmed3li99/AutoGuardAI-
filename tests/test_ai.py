"""Rules, thresholds, Isolation Forest, MOMENT loading/failure, pipeline statuses."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from ai.isolation_forest_baseline import IFModel
from ai.model_registry import ModelRegistry
from ai.moment_inference import MODEL_DIR, MomentScorer, MomentUnavailable
from ai.pipeline import overall_status, run_analysis
from ai.rules_baseline import RuleConfig, run_rules
from ai.thresholds import calibrate, scores_to_alerts
from core.data_sources.files import RecordedFileSource
from core.data_sources.synthetic_demo import SyntheticDemoDataSource
from core.paths import MODELS_DIR
from core.preprocessing.resample import resample_1hz


def frame(n=300, **cols):
    d = {"t_s": np.arange(n, dtype=float)}
    d.update({k: np.full(n, v, dtype=float) if np.isscalar(v) else np.asarray(v, float) for k, v in cols.items()})
    return pd.DataFrame(d)


def test_rule_coolant_requires_duration():
    c = np.full(300, 90.0)
    c[100:105] = 115            # 5 s spike: below 10 s minimum
    c[200:230] = 112            # 30 s: warning
    alerts = run_rules(frame(coolant_temp_c=c))
    assert len(alerts) == 1 and alerts[0].severity == "warning"
    assert (alerts[0].t_start_s, alerts[0].t_end_s) == (200, 229)
    assert alerts[0].values["coolant_temp_c"] == 112


def test_rule_low_voltage_only_with_engine_running():
    v = np.full(300, 11.5)
    rpm = np.zeros(300)
    rpm[150:] = 800
    alerts = run_rules(frame(battery_voltage=v, rpm=rpm))
    assert [a.detector for a in alerts] == ["rule:charging_voltage_low"]
    assert alerts[0].t_start_s == 150


def test_rules_configurable():
    c = np.full(100, 100.0)
    assert run_rules(frame(100, coolant_temp_c=c)) == []
    assert run_rules(frame(100, coolant_temp_c=c), RuleConfig(coolant_attention_c=99)) != []


def test_threshold_calibration_and_alert_merging():
    rng = np.random.default_rng(0)
    thr = calibrate(rng.normal(size=10000), "unit")
    assert thr.attention == pytest.approx(np.quantile(rng.normal(size=10**6), 0.995), abs=0.1)
    s = np.zeros(200)
    s[50:60] = 10
    s[65:70] = 10               # gap 5 s <= merge gap -> one alert
    alerts = scores_to_alerts(frame(200, rpm=1000), s, thr, "model:x", "v1")
    assert len(alerts) == 1 and alerts[0].t_start_s == 50 and alerts[0].t_end_s == 69
    with pytest.raises(ValueError):
        calibrate(np.array([1.0, 2.0]), "too small")


def test_iforest_artifact_loads_and_scores(toyota_sample):
    m = IFModel.load(MODELS_DIR / "iforest")
    assert m.threshold is not None and m.meta["calibrated_on_dataset"] == "toyota_carobd"
    s = RecordedFileSource(toyota_sample).read_session()
    df, _ = resample_1hz(s.data, s.signals)
    sc, contrib = m.score(df)
    assert sc.shape == (len(df),) and np.isfinite(sc).mean() > 0.9
    assert list(contrib.columns) == m.signals


def test_iforest_refuses_tampered_artifact(tmp_path):
    shutil.copytree(MODELS_DIR / "iforest", tmp_path / "if")
    with open(tmp_path / "if" / "iforest.joblib", "ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        IFModel.load(tmp_path / "if")


def test_moment_missing_weights_is_graceful(tmp_path):
    with pytest.raises(MomentUnavailable):
        MomentScorer.load(tmp_path)
    reg = ModelRegistry(iforest_dir=tmp_path / "nope", moment_dir=tmp_path)
    s = SyntheticDemoDataSource().read_session()    # full length: includes injected events
    r = run_analysis(s, reg)
    assert r.detector_status["moment"]["status"] == "unavailable"
    assert r.detector_status["isolation_forest"]["status"] == "unavailable"
    assert any(a.detector.startswith("rule:") for a in r.alerts)     # rules still work
    assert overall_status(r, s) == "Warning"


@pytest.mark.skipif(not (MODEL_DIR / "moment_meta.json").exists(), reason="MOMENT weights not provisioned")
def test_moment_loads_offline_and_scores_cpu():
    import os
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    m = MomentScorer.load(MODEL_DIR, "cpu")
    df = SyntheticDemoDataSource(duration_s=700).read_session().data
    sc, contrib, info = m.score(df)
    assert sc.shape == (700,) and np.isfinite(sc).all() and info["windows"] == 2
    short = df.iloc[:100]                                 # shorter than 512 -> padded + masked
    sc2, _, _ = m.score(short)
    assert sc2.shape == (100,) and np.isfinite(sc2).all()


@pytest.mark.skipif(not (MODEL_DIR / "PROVENANCE.json").exists(), reason="MOMENT weights not provisioned")
def test_moment_weights_match_pinned_hash():
    from core.data_sources.files import sha256_file
    prov = json.loads((MODEL_DIR / "PROVENANCE.json").read_text())
    assert sha256_file(MODEL_DIR / "model.safetensors") == prov["files"]["model.safetensors"]["sha256"]
    assert not (MODEL_DIR / "pytorch_model.bin").exists()


def test_models_report_missing_signals():
    reg = ModelRegistry(enable_moment=False)
    df = pd.DataFrame({"t_s": np.arange(200.0), "rpm": 800.0})
    from core.schema import DataOrigin, NormalizedSession, SessionMetadata, TimeBasis
    s = NormalizedSession(df, SessionMetadata("x", DataOrigin.RECORDED, TimeBasis.RELATIVE_RECORDED))
    r = run_analysis(s, reg)
    assert r.detector_status["isolation_forest"]["status"] == "unavailable"
    assert "Required signals missing" in r.detector_status["isolation_forest"]["reason"]


def test_ml_only_alerts_never_raise_warning_status(toyota_sample):
    from ai.events import Alert, AnalysisResult
    s = RecordedFileSource(toyota_sample).read_session()
    ml = Alert(10, 20, "warning", "model:moment", "ml_anomaly", ["rpm"], "x")
    assert overall_status(AnalysisResult([ml], {}, {}, {}, {}), s) == "Attention"
    rule = Alert(10, 20, "warning", "rule:coolant_high", "heuristic_rule", ["coolant_temp_c"], "x")
    assert overall_status(AnalysisResult([ml, rule], {}, {}, {}, {}), s) == "Warning"


def test_insufficient_data_status():
    from core.schema import DataOrigin, NormalizedSession, SessionMetadata, TimeBasis
    s = NormalizedSession(frame(20, rpm=800), SessionMetadata("x", DataOrigin.RECORDED, TimeBasis.RELATIVE_RECORDED))
    assert overall_status(None, s) == "Insufficient Data"


def test_synthetic_demo_is_labelled_and_deterministic():
    a = SyntheticDemoDataSource().read_session()
    b = SyntheticDemoDataSource().read_session()
    pd.testing.assert_frame_equal(a.data, b.data)
    assert a.meta.origin.value == "synthetic"
    assert any("SYNTHETIC" in n for n in a.meta.notes)
    r = run_analysis(a, ModelRegistry(enable_moment=False))
    assert all(x.origin == "synthetic" for x in r.alerts)
    kinds = {x.detector for x in r.alerts}
    assert {"rule:coolant_high", "rule:charging_voltage_low", "rule:rpm_stuck"} <= kinds
