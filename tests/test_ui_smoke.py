"""UI startup, screens, language switch and the demo flow (offscreen Qt)."""
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
    app = QApplication.instance() or QApplication([])  # noqa: F841 - must stay referenced while Qt runs
    st = AppState(Repository(connect(tmp_path / "ui.sqlite3")))
    w = MainWindow(st)
    yield app, st, w
    w.close()


def _load_demo(app, st):
    from ai.model_registry import ModelRegistry
    from ai.pipeline import run_analysis
    from core.data_sources.synthetic_demo import SyntheticDemoDataSource
    src = SyntheticDemoDataSource()
    s = src.read_session()
    sid = st.repo.save_session(s, src.quality.to_dict())
    st.set_session(s, src.quality.to_dict(), sid)
    st.registry = ModelRegistry(enable_moment=False)
    st._analysis_done((st._generation, sid, run_analysis(s, st.registry)))
    app.processEvents()


def test_starts_on_home_with_four_screens(window):
    _, st, w = window
    assert list(w.pages) == ["home", "trip", "results", "report"]
    assert w.current() == "home"


def test_demo_flow_results_trip_report(window):
    app, st, w = window
    _load_demo(app, st)
    w.navigate("results")
    app.processEvents()
    main = [f for f in st.findings if f.confidence != "low"]
    assert any(f.signal == "coolant_temp_c" and f.severity == "warning" for f in main)
    assert any(f.signal == "battery_voltage" and f.severity == "warning" for f in main)
    assert all("مصطنع" in f.ar["evidence"] for f in main if f.signal in ("coolant_temp_c", "battery_voltage"))
    # "show on chart" moves the Trip screen to the finding
    w.pages["results"]._show(main[0])
    app.processEvents()
    assert w.current() == "trip"
    assert st.engine.position_s >= main[0].t_end_s
    w.navigate("report")
    assert w.pages["report"].b_pdf.isEnabled()


def test_language_switch_rebuilds_rtl(window):
    app, st, w = window
    from PySide6.QtCore import Qt
    st.lang = "en"
    w.toggle_lang()
    assert st.lang == "ar" and app.layoutDirection() == Qt.RightToLeft
    assert w.nav_buttons["home"].text() == "الرئيسية"
    w.toggle_lang()
    assert app.layoutDirection() == Qt.LeftToRight and w.nav_buttons["home"].text() == "Home"
    assert st.repo.get_setting("language") == "en"


def test_live_dialog_with_simulated_adapter(window):
    """Simulated adapter only - NOT a real hardware test."""
    app, st, w = window
    from PySide6.QtCore import QEventLoop, QTimer
    from app.pages.live_dialog import LiveDialog
    d = LiveDialog(st, w)
    d.start(simulated=True)
    loop = QEventLoop()
    d.worker.sig.connected.connect(lambda _: QTimer.singleShot(2500, loop.quit))
    QTimer.singleShot(30000, loop.quit)
    loop.exec()
    assert "SIMULATED" in d.info_lbl.text()
    assert d.cards["rpm"].value.text() not in ("N/A", "مش متاحة", "—")
    d.stop()
    loop2 = QEventLoop()
    d.worker.sig.stopped.connect(loop2.quit)
    QTimer.singleShot(5000, loop2.quit)
    loop2.exec()
    d.close()


def test_results_page_with_a_trip_shows_advanced_button(window):
    """Regression: the Results page referenced an unset `open_advanced` and failed whenever a trip was open."""
    from core.data_sources.files import RecordedFileSource
    from core.paths import SAMPLE_DIR
    app, st, w = window
    st.session = RecordedFileSource(SAMPLE_DIR / "toyota_carobd_long10.csv").read_session()
    page = w.pages["results"]
    page.rebuild()
    texts = [b.text() for b in page.findChildren(type(w.nav_buttons["home"]))]
    assert any(t in ("Advanced analysis", "التحليل المتقدم") for t in texts)
