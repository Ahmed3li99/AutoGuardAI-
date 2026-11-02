"""Offscreen GUI integration of bundled models, asynchronous jobs and exports."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
OUT = ROOT / ".test-tmp" / "advanced-ui"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["AUTOGUARD_HOME"] = str(OUT / "home")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase
from PySide6.QtCore import QEventLoop, QTimer
from app.state import AppState
from app.pages.advanced_dialog import AdvancedDialog
from app.theme import QSS, configure_pyqtgraph
from core.data_sources.files import RecordedFileSource
from core.paths import SAMPLE_DIR
from database.connection import connect
from database.repositories import Repository

app = QApplication([])
# Offscreen Windows does not enumerate system fonts in some sandbox sessions.
for name in ("segoeui.ttf", "segoeuib.ttf", "tahoma.ttf"):
    path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
    if path.exists(): QFontDatabase.addApplicationFont(str(path))
app.setStyleSheet(QSS); configure_pyqtgraph()
state = AppState(Repository(connect(OUT / "ui.sqlite3")))
session = RecordedFileSource(SAMPLE_DIR / "toyota_carobd_long10.csv").read_session()
state.set_session(session, None, state.repo.save_session(session))
dialog = AdvancedDialog(state); dialog.show()
checks = {}
for index, action in [(1, dialog.graph), (2, dialog.forecast), (3, dialog.active)]:
    dialog.tabs.setCurrentIndex(index); action()
    loop = QEventLoop(); timer = QTimer()
    timer.timeout.connect(lambda: loop.quit() if not dialog.busy else None)
    timer.start(50); QTimer.singleShot(60000, loop.quit); loop.exec(); timer.stop()
    app.processEvents()
    if dialog.busy or not dialog.last_output or dialog.last_output["tool"] != {1:"graph",2:"forecast",3:"active"}[index]:
        raise RuntimeError(dialog.status.text() or "Timed out")
    if "اكتمل" not in dialog.status.text() and "Completed" not in dialog.status.text():
        raise RuntimeError(dialog.status.text())
    dialog.grab().save(str(OUT / f"tab-{index}.png"))
    checks[str(index)] = dialog.status.text()
dialog.tabs.setCurrentIndex(4); app.processEvents()
dialog.grab().save(str(OUT / "fusion.png"))
dialog.tabs.setCurrentIndex(5); app.processEvents()
dialog.grab().save(str(OUT / "federation.png"))
dialog.close()
(OUT / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(checks, ensure_ascii=True))
