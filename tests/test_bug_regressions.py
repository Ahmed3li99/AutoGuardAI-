"""Regression tests for the five bugs reported after the v0.2.0 review."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai.events import Alert, AnalysisResult
from ai.explain import findings, verdict
from ai.pipeline import overall_status
from core.schema import DataOrigin, NormalizedSession, SessionMetadata, TimeBasis

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _session(n=600, **cols):
    d = {"t_s": np.arange(n, dtype=float), "rpm": np.full(n, 900.0)}
    d.update({k: np.full(n, v, dtype=float) for k, v in cols.items()})
    return NormalizedSession(pd.DataFrame(d), SessionMetadata("x", DataOrigin.RECORDED, TimeBasis.RELATIVE_RECORDED))


ALL_OK = {k: {"status": "ok"} for k in ("rules", "expected_vs_actual", "isolation_forest", "moment")}


# ---- 1. analysis result attached to the wrong trip --------------------------------------
def test_slow_analysis_is_saved_to_its_own_trip_not_the_one_opened_meanwhile(tmp_path):
    pytest.importorskip("PySide6")
    import threading

    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    import app.state as state_mod
    from app.state import AppState
    from database.connection import connect
    from database.repositories import Repository
    app = QApplication.instance() or QApplication([])  # noqa: F841 - must stay referenced while Qt runs
    st = AppState(Repository(connect(tmp_path / "t.sqlite3")))
    a, b = _session(300), _session(400)
    sid_a = st.repo.save_session(a)
    sid_b = st.repo.save_session(b)
    st.set_session(a, None, sid_a)

    release = threading.Event()
    real = state_mod.run_analysis

    def slow(session, registry, cfg):          # simulate a slow analysis of trip A
        release.wait(10)
        return real(session, registry, cfg)

    state_mod.run_analysis = slow
    try:
        from ai.model_registry import ModelRegistry
        st.registry = ModelRegistry(iforest_dir=tmp_path / "x", moment_dir=tmp_path / "x", nbm_dir=tmp_path / "x")
        st.run_analysis_async()
        st.set_session(b, None, sid_b)         # user opens trip B before A's analysis finishes
        release.set()
        loop = QEventLoop()
        st.history_changed.connect(loop.quit)
        QTimer.singleShot(15000, loop.quit)
        loop.exec()
    finally:
        state_mod.run_analysis = real
    assert st.repo.load_latest_analysis(sid_a) is not None        # saved to A
    assert st.repo.load_latest_analysis(sid_b) is None            # NOT to B
    assert st.session is b and st.result is None                   # B's screen not polluted


# ---- 2. "no issues" while models were not running ----------------------------------------
def test_verdict_is_partial_when_models_did_not_run():
    s = _session()
    status = {"rules": {"status": "ok"}, "expected_vs_actual": {"status": "unavailable"},
              "isolation_forest": {"status": "unavailable"}, "moment": {"status": "unavailable"}}
    r = AnalysisResult([], {}, status, {}, {})
    v = verdict(findings(r, s), r, s)
    assert v["partial"] and v["ran"] == ["rules"]
    assert "No issues found on this trip" not in v["en"]
    assert "Partial" in v["en_sub"] and "fixed-limit rule" in v["en_sub"]
    assert overall_status(r, s) == "Normal (partial check)"
    full = AnalysisResult([], {}, ALL_OK, {}, {})
    assert verdict([], full, s)["en"] == "No issues found on this trip"


# ---- 3. short trip hides a rule warning ------------------------------------------------------
def test_short_trip_warning_takes_priority():
    s = _session(30, coolant_temp_c=120.0)                       # 120 C for 30 s
    from ai.rules_baseline import run_rules
    alerts = run_rules(s.data)
    assert any(a.severity == "warning" for a in alerts)
    r = AnalysisResult(alerts, {}, {"rules": {"status": "ok"},
                                    "expected_vs_actual": {"status": "insufficient_data"}}, {}, {})
    v = verdict(findings(r, s), r, s)
    assert v["level"] == "warning"
    assert "shorter than one minute" in v["en_sub"]
    assert overall_status(r, s) == "Warning"


# ---- 4. OBD time running backwards after reconnect ------------------------------------------
def test_obd_reconnect_keeps_session_clock_monotonic():
    from core.data_sources.obd_optional import READ_ONLY_PIDS, OBDDataSource
    t = {"now": 100.0}
    conns = []

    class Conn:
        def __init__(self):
            self.up = True
            conns.append(self)

        def is_connected(self): return self.up
        def status(self): return "ok"
        def supports(self, c): return c.name == "RPM"
        def close(self): self.up = False
        def query(self, c): return SimpleNamespace(is_null=lambda: False,
                                                   value=SimpleNamespace(to=lambda u: SimpleNamespace(magnitude=850.0)))

    mod = SimpleNamespace(OBD=lambda port=None, fast=False: Conn(), scan_serial=lambda: [],
                          commands=SimpleNamespace(**{n: SimpleNamespace(name=n) for n in READ_ONLY_PIDS}))
    src = OBDDataSource(None, enabled=True, obd_module=mod, clock=lambda: t["now"])
    src.open()
    t["now"] = 110.0
    src.poll_once()                                               # t = 10
    conns[-1].up = False                                          # link drops
    t["now"] = 111.0
    src.poll_once()                                               # reconnect, then read
    times = [r.t_s for r in src.recorded]
    assert times == [10.0, 11.0]                                  # was [10, 1] before the fix
    assert any("Reconnected" in e for e in src.errors)
    assert src.read_session().data["t_s"].is_monotonic_increasing


# ---- 5. a missing time value hides a backwards step -----------------------------------------
def test_missing_time_does_not_hide_unsorted_rows(tmp_path):
    from core.data_sources.files import RecordedFileSource
    p = tmp_path / "t.csv"
    p.write_text("t_s,rpm\n2,800\n,810\n1,820\n")
    src = RecordedFileSource(p)
    s = src.read_session()
    assert s.data["t_s"].tolist() == [1.0, 2.0]
    assert s.duration_s == 1.0
    assert src.quality.non_monotonic_time == 1


def test_alert_helper_kept_for_other_tests():
    a = Alert(0, 1, "attention", "rule:x", "heuristic_rule", ["rpm"], "x")
    assert a.to_dict()["detector"] == "rule:x"


def test_coverage_reports_checks_sensors_and_calibration():
    from ai.explain import coverage
    s = _session(600, coolant_temp_c=88.0)                        # rpm + coolant only
    status = {"rules": {"status": "ok"}, "expected_vs_actual": {"status": "ok", "calibration": "uncalibrated_for_this_data"},
              "isolation_forest": {"status": "unavailable"}}
    rows = {r["en"]: r for r in coverage(AnalysisResult([], {}, status, {}, {}), s, None)}
    assert rows["Checks that ran"]["value_en"].startswith("2 of 3") and rows["Checks that ran"]["ok"] is False
    assert rows["Sensors available"]["value_en"].startswith("2 of 8") and not rows["Sensors available"]["ok"]
    assert rows["Model fit for this car"]["ok"] is False and "different car" in rows["Model fit for this car"]["value_en"]
    assert rows["Trip long enough for the models?"]["ok"] is True
