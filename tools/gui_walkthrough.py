"""Drive the real GUI through the demo flow and save a screenshot of every screen
(used to verify the UI renders real data).

Usage:  python tools/gui_walkthrough.py <out_dir> [ar|en] [sample-file]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    out = Path(sys.argv[1])
    lang = sys.argv[2] if len(sys.argv) > 2 else "ar"
    sample = sys.argv[3] if len(sys.argv) > 3 else None
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
    from app.icons import app_icon
    app.setWindowIcon(app_icon())
    configure_pyqtgraph()
    repo = Repository(connect())
    repo.set_setting("language", lang)
    state = AppState(repo)
    win = MainWindow(state)
    win.resize(1400, 900)
    win.show()

    def pump(ms: int) -> None:
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def shot(name: str) -> None:
        pump(400)
        win.grab().save(str(out / f"{name}.png"))

    shot("01_home_empty")
    loop = QEventLoop()
    state.analysis_changed.connect(lambda: loop.quit() if state.result is not None else None)
    state.error.connect(lambda m: (print("ERROR", m), loop.quit()))
    QTimer.singleShot(600_000, loop.quit)
    state.load_source_async(RecordedFileSource(sample) if sample else SyntheticDemoDataSource())
    win.navigate("results")
    loop.exec()
    shot("02_results")
    page = win.pages["results"]
    page.scroll.verticalScrollBar().setValue(450)
    shot("03_results_scrolled")
    if state.findings:
        f = state.findings[0]
        page._show(f)
        shot("04_trip_focus")
    win.navigate("trip")
    state.stop_replay()
    state.set_speed(16.0)
    state.play()
    pump(2500)
    state.pause()
    shot("05_trip_playing")
    win.navigate("report")
    shot("06_report")
    win.navigate("home")
    shot("07_home_history")
    print("lang", lang, "findings", [(f.confidence, f.severity, f.en["title"]) for f in state.findings])
    win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
