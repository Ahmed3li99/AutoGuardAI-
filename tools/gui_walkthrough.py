"""Drive the real GUI through the demo flow and save a screenshot of every page.
Used to verify the UI renders real data (not for marketing). Output goes to the
directory given as the first argument.

Usage:  python tools/gui_walkthrough.py <out_dir> [sample-file]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AUTOGUARD_HOME", str(out / "home"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    from app.main_window import MainWindow
    from app.state import AppState
    from app.theme import QSS, configure_pyqtgraph
    from core.data_sources.files import RecordedFileSource
    from core.data_sources.synthetic_demo import SyntheticDemoDataSource
    from database.connection import connect
    from database.repositories import Repository

    app = QApplication([])
    app.setStyleSheet(QSS)
    configure_pyqtgraph()
    state = AppState(Repository(connect()))
    win = MainWindow(state)
    win.resize(1400, 880)
    win.show()

    def pump(ms: int) -> None:
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def wait_analysis(timeout_ms: int = 600_000) -> None:
        loop = QEventLoop()
        state.analysis_changed.connect(loop.quit)
        state.error.connect(lambda m: (print("ERROR", m), loop.quit()))
        QTimer.singleShot(timeout_ms, loop.quit)
        loop.exec()

    sample = sys.argv[2] if len(sys.argv) > 2 else None
    src = RecordedFileSource(sample) if sample else SyntheticDemoDataSource()
    s = src.read_session()
    sid = state.repo.save_session(s, src.quality.to_dict())
    # import page: show the same source through the real UI path
    page = win.pages["Data Import & Quality"]
    if sample:
        page._load_path(Path(sample))
        pump(2500)
    else:
        page.load_synthetic()
    win.navigate("Data Import & Quality")
    pump(400)
    win.grab().save(str(out / "01_import.png"))

    state.set_session(s, src.quality.to_dict(), sid)
    state.run_analysis_async()
    wait_analysis()
    pump(300)
    state.set_speed(16.0)
    state.play()
    pump(2500)
    state.pause()
    state.seek(min(s.duration_s * 0.95, s.duration_s))
    win.navigate("Replay")
    win.pages["Replay"]._rebuild_plots()
    pump(500)
    win.grab().save(str(out / "02_replay.png"))
    for i, name in enumerate(["Overview", "AI & Anomalies", "Events & Alerts", "Trips & History", "Reports",
                              "Settings"], start=3):
        win.navigate(name)
        if name == "AI & Anomalies" and state.result and state.result.alerts:
            ai = win.pages[name]
            ai.alert_tbl.selectRow(len(state.result.alerts) - 1)
        pump(500)
        win.grab().save(str(out / f"{i:02d}_{name.split()[0].lower()}.png"))
    print("detectors:", {k: v.get("status") for k, v in state.result.detector_status.items()})
    print("alerts:", len(state.result.alerts))
    win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
