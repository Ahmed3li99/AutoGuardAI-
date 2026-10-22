"""Reports: offline PDF / CSV export of the current session."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from app.state import AppState, Worker
from app.widgets.common import h1, muted
from reporting.csv_export import export_alerts_csv, export_session_csv
from reporting.pdf_export import build_pdf_report


class ReportsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(h1("Reports"))
        lay.addWidget(muted("Exports are generated locally from the saved session and its latest analysis run, "
                            "so they match what the app shows. Synthetic sessions are labelled in every export."))
        self.cur = QLabel("No session loaded.")
        lay.addWidget(self.cur)
        row = QHBoxLayout()
        self.b_pdf = QPushButton("Export PDF report…")
        self.b_pdf.setObjectName("primary")
        self.b_pdf.clicked.connect(self.export_pdf)
        self.b_alerts = QPushButton("Export alerts CSV…")
        self.b_alerts.clicked.connect(self.export_alerts)
        self.b_data = QPushButton("Export normalized data CSV…")
        self.b_data.clicked.connect(self.export_data)
        for w in (self.b_pdf, self.b_alerts, self.b_data):
            row.addWidget(w)
        row.addStretch()
        lay.addLayout(row)
        self.last = QLabel("")
        self.last.setWordWrap(True)
        lay.addWidget(self.last)
        lay.addStretch()
        state.session_changed.connect(self.refresh)
        state.analysis_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        s = self.state.session
        on = s is not None
        for b in (self.b_pdf, self.b_data):
            b.setEnabled(on)
        self.b_alerts.setEnabled(on and self.state.result is not None)
        if on:
            self.cur.setText(f"Current: {s.meta.source_name} ({s.meta.origin.label}) · session "
                             f"#{self.state.session_id} · analysis: "
                             f"{'run #%s' % self.state.run_id if self.state.result else 'not run'}")

    def _ask(self, title: str, default: str, filt: str) -> str | None:
        stem = Path(self.state.session.meta.source_name.replace(":", "_")).stem
        path, _ = QFileDialog.getSaveFileName(self, title, f"{stem}_{default}", filt)
        return path or None

    def _done(self, path) -> None:
        self.state.busy_changed.emit(False, f"Saved {path}")
        self.last.setText(f"Last export: {path}")
        if QMessageBox.question(self, "Export complete", f"Saved:\n{path}\n\nOpen it now?") == QMessageBox.Yes:
            os.startfile(str(path))  # noqa: S606 - local file the user just created

    def export_pdf(self) -> None:
        path = self._ask("Export PDF", "report.pdf", "PDF (*.pdf)")
        if not path:
            return
        st = self.state
        self.state.busy_changed.emit(True, "Building PDF…")
        w = Worker(build_pdf_report, path, st.session, st.quality, st.result, st.session_id, st.vehicle_profile())
        w.signals.finished.connect(self._done)
        w.signals.failed.connect(lambda m: (self.state.busy_changed.emit(False, ""), self.state.error.emit(m)))
        self.state.pool.start(w)

    def export_alerts(self) -> None:
        path = self._ask("Export alerts", "alerts.csv", "CSV (*.csv)")
        if path:
            self._done(export_alerts_csv(self.state.session, self.state.result, path, self.state.session_id))

    def export_data(self) -> None:
        path = self._ask("Export data", "data.csv", "CSV (*.csv)")
        if path:
            self._done(export_session_csv(self.state.session, path))
