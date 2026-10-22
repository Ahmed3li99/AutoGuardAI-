"""Events & Alerts across all saved sessions (latest analysis run of each)."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.state import AppState
from app.widgets.common import fill_table, h1, muted, readonly_table


class EventsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Events & Alerts"))
        lay.addWidget(muted("All alerts from the latest analysis run of every saved session. 'origin' shows "
                            "whether the alert came from recorded or synthetic data. Double-click to open."))
        row = QHBoxLayout()
        self.f_sev = QComboBox()
        self.f_sev.addItem("All severities", None)
        self.f_sev.addItem("Warning", "warning")
        self.f_sev.addItem("Attention", "attention")
        self.f_kind = QComboBox()
        self.f_kind.addItem("All detectors", None)
        self.f_kind.addItem("Heuristic rules", "heuristic_rule")
        self.f_kind.addItem("ML anomaly", "ml_anomaly")
        self.f_session = QComboBox()
        for w in (self.f_sev, self.f_kind, self.f_session):
            w.currentIndexChanged.connect(self.refresh)
        b_exp = QPushButton("Export filtered CSV…")
        b_exp.clicked.connect(self.export)
        for w in (QLabel("Filter:"), self.f_sev, self.f_kind, self.f_session):
            row.addWidget(w)
        row.addStretch()
        row.addWidget(b_exp)
        lay.addLayout(row)
        self.table = readonly_table()
        self.table.doubleClicked.connect(self._open)
        lay.addWidget(self.table, 1)
        state.history_changed.connect(self._reload_sessions)
        state.analysis_changed.connect(self.refresh)
        self._reload_sessions()

    def _reload_sessions(self) -> None:
        cur = self.f_session.currentData()
        self.f_session.blockSignals(True)
        self.f_session.clear()
        self.f_session.addItem("All sessions", None)
        for r in self.state.repo.list_sessions().itertuples():
            self.f_session.addItem(f"#{r.id} {r.source_name}", int(r.id))
        idx = self.f_session.findData(cur)
        self.f_session.setCurrentIndex(max(idx, 0))
        self.f_session.blockSignals(False)
        self.refresh()

    def _frame(self):
        df = self.state.repo.list_alerts(self.f_session.currentData(), self.f_sev.currentData(),
                                         self.f_kind.currentData())
        return df.drop(columns=["values_json"]).rename(columns={"signals_json": "signals"})

    def refresh(self) -> None:
        self._df = self._frame()
        fill_table(self.table, self._df, float_fmt="{:.4g}")

    def export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export alerts", "alerts.csv", "CSV (*.csv)")
        if path:
            self._df.to_csv(path, index=False)
            self.state.busy_changed.emit(False, f"Exported {len(self._df)} alerts to {path}")

    def _open(self, index) -> None:
        sid = int(self._df.iloc[index.row()]["session_id"])
        self.state.open_saved_session(sid)
        self.window().navigate("AI & Anomalies")
