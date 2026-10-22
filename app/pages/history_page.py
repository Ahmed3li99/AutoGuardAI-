"""Trips & History: reopen or delete saved sessions."""
from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from app.state import AppState
from app.widgets.common import fill_table, h1, muted, readonly_table


class HistoryPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Trips & History"))
        lay.addWidget(muted("Sessions saved in the local SQLite database. Deleting removes readings, analysis "
                            "runs and alerts of that session from this computer."))
        self.table = readonly_table()
        self.table.doubleClicked.connect(lambda _: self.open_selected())
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        b_open = QPushButton("Open")
        b_open.setObjectName("primary")
        b_open.clicked.connect(self.open_selected)
        b_del = QPushButton("Delete selected")
        b_del.clicked.connect(self.delete_selected)
        b_all = QPushButton("Delete ALL local history")
        b_all.setObjectName("danger")
        b_all.clicked.connect(self.delete_all)
        for w in (b_open, b_del):
            row.addWidget(w)
        row.addStretch()
        row.addWidget(b_all)
        lay.addLayout(row)
        state.history_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self._df = self.state.repo.list_sessions()
        show = self._df.drop(columns=["signals_json"]).rename(columns={"n_alerts": "alerts (latest run)"})
        fill_table(self.table, show, float_fmt="{:.0f}")

    def _selected_id(self) -> int | None:
        rows = self.table.selectionModel().selectedRows()
        return int(self._df.iloc[rows[0].row()]["id"]) if rows else None

    def open_selected(self) -> None:
        sid = self._selected_id()
        if sid is not None:
            self.state.open_saved_session(sid)
            self.window().navigate("Overview")

    def delete_selected(self) -> None:
        sid = self._selected_id()
        if sid is None:
            return
        if QMessageBox.question(self, "Delete session", f"Delete session #{sid} and its alerts?") == QMessageBox.Yes:
            self.state.repo.delete_session(sid)
            if self.state.session_id == sid:
                self.state.session_id = None
            self.state.history_changed.emit()

    def delete_all(self) -> None:
        if QMessageBox.question(self, "Delete all history",
                                "Permanently delete ALL saved sessions, readings and alerts on this computer?") \
                == QMessageBox.Yes:
            self.state.repo.delete_all_history()
            self.state.session_id = None
            self.state.history_changed.emit()
