"""Scenario lab and 'what if?'."""
import os

import numpy as np
import pytest

from ai.model_registry import ModelRegistry
from ai.scenarios import run_scenario, what_if_normal
from core.analytics.injection import InjectedEvent, inject
from core.data_sources.files import RecordedFileSource
from core.data_sources.synthetic_demo import SyntheticDemoDataSource
from core.paths import SAMPLE_DIR

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REG = dict(enable_moment=False)


@pytest.fixture(scope="module")
def trip():
    return RecordedFileSource(SAMPLE_DIR / "toyota_carobd_long10.csv").read_session()


def test_new_injection_kinds():
    import pandas as pd
    df = pd.DataFrame({"t_s": np.arange(100.0), "battery_voltage": 13.8, "coolant_temp_c": 88.0})
    d = inject(df, [InjectedEvent("voltage_decline", "battery_voltage", 10, 60, "t", 11.6)])
    assert d["battery_voltage"].iloc[10] == pytest.approx(13.8) and d["battery_voltage"].iloc[59] == pytest.approx(11.6)
    assert (np.diff(d["battery_voltage"].iloc[10:60]) <= 0).all()
    d = inject(df, [InjectedEvent("step_jump", "coolant_temp_c", 20, 30, "t", 15)])
    assert d["coolant_temp_c"].iloc[25] == 103.0 and d["coolant_temp_c"].iloc[30] == 88.0
    d = inject(df, [InjectedEvent("coolant_overheat", "coolant_temp_c", 0, 80, "t")])
    assert d["coolant_temp_c"].max() == pytest.approx(112.0)
    assert np.abs(np.diff(d["coolant_temp_c"])).max() < 3          # no physically impossible jump at the end


def test_gradual_decline_found_earlier_by_model_than_by_rule(trip):
    ev = InjectedEvent("voltage_decline", "battery_voltage", 1000, 1600, "lab", 11.6)
    r = run_scenario(trip, [ev], ModelRegistry(**REG))
    pd_ = r.per_detector
    assert pd_["rules"]["caught"] == [True] and pd_["expected_vs_actual"]["caught"] == [True]
    assert pd_["expected_vs_actual"]["delay_s"] < pd_["rules"]["delay_s"]
    assert pd_["moment"]["ran"] is False
    assert r.modified.meta.origin.value == "synthetic"                # never presented as real data
    assert any(f.signal == "battery_voltage" for f in r.diff["new"])


def test_sensor_jump_caught_by_consistency(trip):
    r = run_scenario(trip, [InjectedEvent("step_jump", "coolant_temp_c", 1200, 1300, "lab", 15)], ModelRegistry(**REG))
    assert r.per_detector["consistency"]["caught"] == [True] and r.per_detector["consistency"]["delay_s"] == 0
    assert all(f.category == "data" for f in r.diff["new"] if f.signal == "coolant_temp_c")


def test_pre_existing_alerts_are_not_credited(trip):
    """A scenario identical to the original must not count as 'caught' by anyone."""
    r = run_scenario(trip, [InjectedEvent("step_jump", "coolant_temp_c", 1200, 1300, "lab", 0.0)], ModelRegistry(**REG))
    assert not any(v["caught"][0] for v in r.per_detector.values())


def test_what_if_removes_only_the_related_finding():
    demo = SyntheticDemoDataSource().read_session()
    w = what_if_normal(demo, "battery_voltage", 1150, 1350, ModelRegistry(**REG))
    assert [f.signal for f in w.diff["gone"] if f.confidence != "low"] == ["battery_voltage"]
    kept = {f.signal for f in w.diff["kept"] if f.confidence != "low"}
    assert "coolant_temp_c" in kept                                      # independent problem remains
    with pytest.raises(ValueError):
        what_if_normal(demo, "throttle_pct", 0, 10, ModelRegistry(**REG))  # no expected-value model for it


def test_lab_dialog_runs(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    from app.main_window import MainWindow
    from app.pages.lab_dialog import LabDialog
    from app.state import AppState
    from database.connection import connect
    from database.repositories import Repository
    app = QApplication.instance() or QApplication([])  # noqa: F841 - must stay referenced while Qt runs
    repo = Repository(connect(tmp_path / "l.sqlite3"))
    repo.set_setting("moment_enabled", False)
    st = AppState(repo)
    w = MainWindow(st)
    src = SyntheticDemoDataSource()
    s = src.read_session()
    st.set_session(s, src.quality.to_dict(), repo.save_session(s, src.quality.to_dict()))
    d = LabDialog(st, w)
    d.kind.setCurrentIndex(d.kind.findData("voltage_drop"))
    d.start.setValue(300)
    d.dur.setValue(60)
    loop = QEventLoop()
    d.run_fault()

    def poll():
        loop.quit() if d.b_run.isEnabled() else QTimer.singleShot(200, poll)
    QTimer.singleShot(200, poll)
    QTimer.singleShot(120000, loop.quit)
    loop.exec()
    assert "✓" in "".join(d.det_grid.itemAt(i).widget().text() for i in range(d.det_grid.count()))
    d.close()
    w.close()
