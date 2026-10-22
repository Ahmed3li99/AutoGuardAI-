"""Main window: sidebar navigation + page stack + status bar."""
from __future__ import annotations

from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox, QStackedWidget,
                               QWidget)

from app import theme
from app.pages.ai_page import AIPage
from app.pages.dashboard_page import DashboardPage
from app.pages.events_page import EventsPage
from app.pages.history_page import HistoryPage
from app.pages.import_page import ImportPage
from app.pages.replay_page import ReplayPage
from app.pages.reports_page import ReportsPage
from app.pages.settings_page import SettingsPage
from app.state import AppState


class MainWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setWindowTitle("AutoGuard AI — Offline Vehicle Telemetry Prototype")
        self.resize(1400, 880)
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(210)
        self.stack = QStackedWidget()
        self.pages = {
            "Overview": DashboardPage(state),
            "Data Import & Quality": ImportPage(state),
            "Replay": ReplayPage(state),
            "AI & Anomalies": AIPage(state),
            "Events & Alerts": EventsPage(state),
            "Trips & History": HistoryPage(state),
            "Reports": ReportsPage(state),
            "Settings": SettingsPage(state),
        }
        for name, page in self.pages.items():
            self.nav.addItem(name)
            page.setContentsMargins(12, 8, 12, 8)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        lay.addWidget(self.nav)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.mode_lbl = QLabel("")
        self.statusBar().addPermanentWidget(self.mode_lbl)
        state.busy_changed.connect(lambda _on, msg: self.statusBar().showMessage(msg, 8000))
        state.error.connect(lambda msg: QMessageBox.warning(self, "AutoGuard AI", msg))
        state.session_changed.connect(self._mode)
        self.nav.setCurrentRow(1 if state.session is None else 0)
        self._mode()

    def navigate(self, name: str) -> None:
        self.nav.setCurrentRow(list(self.pages).index(name))

    def _mode(self) -> None:
        s = self.state.session
        if s is None:
            self.mode_lbl.setText("Offline · no session")
            return
        c = theme.ORIGIN_COLORS[s.meta.origin.value]
        self.mode_lbl.setText(f"Offline · <b style='color:{c}'>{s.meta.origin.label}</b> · Live OBD-II: "
                              "not connected (future)")
