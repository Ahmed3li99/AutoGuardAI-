"""Basic UI startup and page wiring (offscreen Qt)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def window(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from app.main_window import MainWindow
    from app.state import AppState
    from database.connection import connect
    from database.repositories import Repository
    app = QApplication.instance() or QApplication([])
    st = AppState(Repository(connect(tmp_path / "ui.sqlite3")))
    w = MainWindow(st)
    yield app, st, w
    w.close()


def test_starts_with_all_pages(window):
    _, st, w = window
    assert set(w.pages) == {"Overview", "Data Import & Quality", "Replay", "AI & Anomalies", "Events & Alerts",
                            "Trips & History", "Reports", "Settings"}
    assert "no session" in w.mode_lbl.text()


def test_synthetic_session_flows_through_pages(window):
    app, st, w = window
    from ai.model_registry import ModelRegistry
    from ai.pipeline import run_analysis
    from core.data_sources.synthetic_demo import SyntheticDemoDataSource
    src = SyntheticDemoDataSource()
    s = src.read_session()
    sid = st.repo.save_session(s, src.quality.to_dict())
    st.set_session(s, src.quality.to_dict(), sid)
    st.registry = ModelRegistry(enable_moment=False)
    st._analysis_done(run_analysis(s, st.registry))
    app.processEvents()
    assert "Synthetic Demo" in w.mode_lbl.text()
    dash = w.pages["Overview"]
    assert dash.status_badge.text() == "Warning"
    assert dash.cards["mass_air_flow_g_per_s" if "mass_air_flow_g_per_s" in dash.cards else "rpm"]
    st.seek(1400)
    app.processEvents()
    assert dash.cards["rpm"].value.text() not in ("N/A", "—")
    assert w.pages["Events & Alerts"].table.rowCount() == len(st.result.alerts)
    assert w.pages["Trips & History"].table.rowCount() == 1


def test_obd_status_is_honest(window):
    _, _, w = window
    txt = w.pages["Settings"].h_status.text()
    assert "not connected" in txt.lower()
