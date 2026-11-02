"""Expected-vs-actual model, plain-language findings, and read-only car identification."""
from types import SimpleNamespace

import numpy as np
import pytest

from ai.explain import count_ar, display_name, findings, group_findings, verdict
from ai.model_registry import ModelRegistry
from ai.normal_behavior import NormalBehaviorModel
from ai.pipeline import run_analysis
from core.analytics.injection import InjectedEvent, inject
from core.data_sources.files import RecordedFileSource
from core.data_sources.synthetic_demo import SyntheticDemoDataSource
from core.paths import MODELS_DIR
from core.preprocessing.resample import resample_1hz

NBM_DIR = MODELS_DIR / "normal_behavior"
needs_nbm = pytest.mark.skipif(not (NBM_DIR / "nbm.joblib").exists(), reason="model not trained")


def _trip(sample):
    s = RecordedFileSource(sample).read_session()
    return resample_1hz(s.data, s.signals)[0]


@needs_nbm
def test_nbm_loads_and_predicts_expected_values(toyota_sample):
    m = NormalBehaviorModel.load(NBM_DIR)
    assert m.meta["calibrated_on_dataset"] == "toyota_carobd"
    df = _trip(toyota_sample)
    pred = m.predict(df)
    # On a normal held-out trip the expected coolant should be close to the measured one most of the time
    err = (pred["coolant_temp_c__expected"] - df["coolant_temp_c"]).abs()
    assert err.median() < 3.0


@needs_nbm
def test_nbm_flags_injected_overheat_with_expected_value(toyota_sample):
    m = NormalBehaviorModel.load(NBM_DIR)
    df = _trip(toyota_sample)
    t0 = float(df["t_s"].iloc[400])
    hot = inject(df, [InjectedEvent("coolant_overheat", "coolant_temp_c", t0, t0 + 200, "test")])
    al = [a for a in m.alerts(hot) if a.signals == ["coolant_temp_c"]]
    assert al and al[0].t_start_s >= t0
    assert al[0].values["coolant_temp_c__expected"] < 100       # expected stays normal
    assert m.alerts(df) == [] or all(a.signals != ["coolant_temp_c"] for a in m.alerts(df))


@needs_nbm
def test_nbm_is_quiet_on_normal_samples():
    from core.paths import SAMPLE_DIR
    for name in ("toyota_carobd_ufpe7.csv", "toyota_carobd_idle3.csv"):
        assert NormalBehaviorModel.load(NBM_DIR).alerts(_trip(SAMPLE_DIR / name)) == []


@needs_nbm
def test_demo_findings_are_explained_and_grouped():
    s = SyntheticDemoDataSource().read_session()
    r = run_analysis(s, ModelRegistry(enable_moment=False))
    fs = findings(r, s)
    assert len(group_findings(r.alerts)) == len(fs) < len(r.alerts)       # grouping reduces clutter
    top = fs[0]
    assert top.severity == "warning" and top.confidence == "high"
    for lang in ("ar", "en"):
        t = top.text(lang)
        assert all(t[k] for k in ("title", "what", "normal", "meaning", "action", "evidence"))
    # injected faults say so; findings from the real part of the trip say the opposite
    for f in fs:
        injected = any(e["start_s"] <= f.t_end_s and e["end_s"] >= f.t_start_s for e in s.meta.injected_events)
        assert ("artificial fault we injected" in f.en["evidence"]) == injected
        assert ("real recorded trip" in f.en["evidence"]) != injected
    assert verdict(fs, r, s)["level"] == "warning"


def test_verdict_ok_and_arabic_counting(toyota_sample):
    s = RecordedFileSource(toyota_sample).read_session()
    from ai.events import AnalysisResult
    empty = AnalysisResult([], {}, {}, {}, {})
    assert verdict([], empty, s)["level"] == "ok"
    assert count_ar(1, "مشكلة", "مشكلتين", "مشاكل", "مشكلة") == "مشكلة واحدة"
    assert count_ar(2, "مشكلة", "مشكلتين", "مشاكل", "مشكلة") == "مشكلتين"
    assert count_ar(5, "ملاحظة", "ملاحظتين", "ملاحظات", "ملاحظة") == "5 ملاحظات"
    assert count_ar(14, "ملاحظة", "ملاحظتين", "ملاحظات", "ملاحظة") == "14 ملاحظة"
    assert display_name("synthetic:real_trip_faults", "en").startswith("Demo")
    assert display_name("my_file.csv", "ar") == "my_file.csv"


def test_obd_identify_and_read_dtcs_read_only():
    """MOCKED: VIN + stored trouble codes are read; codes are never cleared."""
    from core.data_sources.obd_optional import READ_ONLY_PIDS, OBDDataSource
    queried = []

    class Conn:
        def is_connected(self): return True
        def status(self): return "Car Connected"
        def port_name(self): return "COM5"
        def protocol_name(self): return "ISO 15765-4 (CAN 11/500)"
        def supports(self, c): return c.name in ("RPM", "VIN")
        def close(self): pass

        def query(self, c):
            queried.append(c.name)
            return SimpleNamespace(is_null=lambda: False,
                                   value={"VIN": b"JTDBT923771234567", "GET_DTC": [("P0118", "ECT circuit high")]}.get(
                                       c.name, SimpleNamespace(to=lambda u: SimpleNamespace(magnitude=800.0))))

    names = list(READ_ONLY_PIDS) + ["VIN", "GET_DTC", "CLEAR_DTC"]
    mod = SimpleNamespace(OBD=lambda port=None, fast=False: Conn(), scan_serial=lambda: ["COM5"],
                          commands=SimpleNamespace(**{n: SimpleNamespace(name=n) for n in names}))
    src = OBDDataSource(None, enabled=True, obd_module=mod)
    src.open()
    info = src.identify()
    assert info == {"port": "COM5", "protocol": "ISO 15765-4 (CAN 11/500)", "vin": "JTDBT923771234567"}
    assert src.read_dtcs() == [("P0118", "ECT circuit high")]
    assert "CLEAR_DTC" not in queried


def test_simulated_adapter_session_is_labelled_synthetic():
    from core.data_sources.obd_optional import OBDDataSource
    from core.data_sources.obd_simulator import simulated_obd_module
    clock = iter(np.arange(0, 100, 0.5)).__next__
    src = OBDDataSource(None, enabled=True, obd_module=simulated_obd_module(clock=clock), simulated=True,
                        clock=clock)
    src.open()
    for _ in range(5):
        src.poll_once()
    s = src.read_session()
    assert s.meta.origin.value == "synthetic" and "SIMULATED" in s.meta.notes[0]
    assert len(s.data) == 5 and s.data["rpm"].notna().all()
