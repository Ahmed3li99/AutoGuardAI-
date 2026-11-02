"""Main window: top bar with 4 screens, language toggle and settings."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QStackedWidget, QVBoxLayout, QWidget)

from app import icons, theme
from app.i18n import pick, set_lang, tr
from app.pages.ai_page import AIPage
from app.pages.home_page import HomePage
from app.pages.import_page import ImportPage
from app.pages.live_dialog import LiveDialog
from app.pages.report_page import ReportPage
from app.pages.results_page import ResultsPage
from app.pages.settings_page import SettingsPage
from app.pages.trip_page import TripPage
from app.state import AppState
from app.widgets.common import Pill, button

SCREENS = ["home", "trip", "results", "report"]


NAV_ICONS = {"home": "home", "trip": "trip", "results": "results", "report": "report"}

class MainWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.resize(1360, 880)
        state.error.connect(lambda msg: QMessageBox.warning(self, tr("error"), msg))
        state.busy_changed.connect(self._busy)
        state.session_changed.connect(self._mode)
        self.build()

    # ---- construction (re-run on language change) ---------------------------
    def build(self) -> None:
        set_lang(self.state.lang)
        QApplication.instance().setLayoutDirection(Qt.RightToLeft if self.state.lang == "ar" else Qt.LeftToRight)
        self.setWindowTitle(tr("app_title"))
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(22, 0, 22, 0)
        hl.setSpacing(8)
        mark = QLabel()
        mark.setPixmap(icons.pixmap_file(icons.logo_path(64), 34))
        hl.addWidget(mark)
        brand = QLabel("AutoGuard")
        brand.setObjectName("brand")
        hl.addWidget(brand)
        ai = QLabel("AI")
        ai.setObjectName("brandAi")
        hl.addWidget(ai, 0, Qt.AlignVCenter)
        hl.addSpacing(28)
        self.nav_group = QButtonGroup(self)
        self.nav_buttons = {}
        for key in SCREENS:
            b = button(tr("nav_" + key), "nav")
            b.setIcon(icons.icon(NAV_ICONS[key], theme.ON_NAVY_MUTED, "#FFFFFF", 18, pad=5))
            b.setIconSize(QSize(28, 18))
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self.navigate(k))
            self.nav_group.addButton(b)
            self.nav_buttons[key] = b
            hl.addWidget(b)
        hl.addStretch()
        self.mode_pill = Pill("")
        hl.addWidget(self.mode_pill)
        offline = Pill(tr("offline"), "#7EE2B8", "#0F2E2A")
        offline.setToolTip(pick("كل التحليل بيحصل على جهازك، من غير نت", "All analysis runs on this PC, no internet"))
        hl.addWidget(offline)
        lang_b = button(tr("lang_toggle"), "onNavy", self.toggle_lang)
        lang_b.setIcon(icons.icon("globe", theme.ON_NAVY, size=16))
        hl.addWidget(lang_b)
        set_b = button(tr("settings"), "onNavy", self.open_settings)
        set_b.setIcon(icons.icon("settings", theme.ON_NAVY, size=16))
        hl.addWidget(set_b)
        root.addWidget(header)

        self.stack = QStackedWidget()
        self.pages = {
            "home": HomePage(self.state, self.open_live, self.open_advanced_import, self.open_cars),
            "trip": TripPage(self.state),
            "results": ResultsPage(self.state, self.open_tech, self.open_lab, self.open_advanced),
            "report": ReportPage(self.state),
        }
        for key in SCREENS:
            self.stack.addWidget(self.pages[key])
        root.addWidget(self.stack, 1)
        old = self.centralWidget()
        self.setCentralWidget(central)
        if old is not None:
            old.deleteLater()
        self.navigate("results" if self.state.session is not None else "home")
        self._mode()

    def navigate(self, key: str) -> None:
        key = key.lower()
        if key not in self.pages:
            key = {"overview": "results", "replay": "trip", "ai & anomalies": "results"}.get(key, "home")
        self.stack.setCurrentWidget(self.pages[key])
        self.nav_buttons[key].setChecked(True)

    def current(self) -> str:
        return next(k for k, p in self.pages.items() if p is self.stack.currentWidget())

    def toggle_lang(self) -> None:
        self.state.lang = "en" if self.state.lang == "ar" else "ar"
        self.state.repo.set_setting("language", self.state.lang)
        cur = self.current()
        self.build()
        self.navigate(cur)

    # ---- dialogs ----------------------------------------------------------------
    def _dialog(self, title: str, widget: QWidget, w: int = 1100, h: int = 760) -> QDialog:
        d = QDialog(self)
        d.setAttribute(Qt.WA_DeleteOnClose)       # its pages must stop listening to state
        d.setWindowTitle(title)
        d.resize(w, h)
        lay = QVBoxLayout(d)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.addWidget(widget)
        return d

    def open_settings(self) -> None:
        self._dialog(tr("settings"), SettingsPage(self.state), 820, 760).exec()

    def open_tech(self) -> None:
        self._dialog(tr("tech_details"), AIPage(self.state)).exec()

    def open_advanced_import(self) -> None:
        self._dialog(tr("advanced_import"), ImportPage(self.state), 1200, 820).exec()

    def open_lab(self) -> None:
        from app.pages.lab_dialog import LabDialog
        d = LabDialog(self.state, self)
        d.setAttribute(Qt.WA_DeleteOnClose)
        d.exec()

    def open_advanced(self) -> None:
        from app.pages.advanced_dialog import AdvancedDialog
        d = AdvancedDialog(self.state, self)
        d.setAttribute(Qt.WA_DeleteOnClose)
        d.exec()

    def open_cars(self) -> None:
        from app.pages.cars_dialog import CarsDialog
        d = CarsDialog(self.state, self)
        d.setAttribute(Qt.WA_DeleteOnClose)
        d.exec()
        self.state.history_changed.emit()      # refresh the car card + trip list

    def open_live(self) -> None:
        d = LiveDialog(self.state, self)
        d.setAttribute(Qt.WA_DeleteOnClose)
        d.exec()

    # ---- status -------------------------------------------------------------------
    def _busy(self, on: bool, msg: str) -> None:
        text = tr(msg) if msg in ("loading", "analysing") else msg
        if msg.startswith("done:"):
            n = int(msg[5:])
            text = pick(f"✓ اتحللت الرحلة — {n} ملاحظة/مشكلة", f"✓ Trip analysed — {n} finding(s)")
        self.statusBar().showMessage(text, 0 if on else 6000)

    def _mode(self) -> None:
        s = self.state.session
        if not hasattr(self, "mode_pill"):
            return
        if s is None:
            self.mode_pill.set(tr("no_trip"), theme.ON_NAVY_MUTED, theme.NAVY_2)
            return
        o = s.meta.origin.value
        # tinted pills that read on the navy header
        fg, bg = {"recorded": ("#9CC0FF", "#16284A"), "synthetic": ("#C9B8FF", "#251B4D"),
                  "live_obd": ("#7EE2B8", "#0F2E2A")}[o]
        self.mode_pill.set(tr("origin_" + o), fg, bg)
