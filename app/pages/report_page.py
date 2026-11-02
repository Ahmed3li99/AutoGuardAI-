"""Report: export PDF (Arabic or English) and CSV files."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget

from app.i18n import lang, pick, tr
from app.state import AppState, Worker
from app.widgets.common import Card, button, h1, muted, wrap
from ai.explain import display_name, findings_frame
from reporting.csv_export import export_session_csv
from reporting.pdf_export import build_pdf_report


class ReportPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.setContentsMargins(36, 28, 36, 28)
        lay.setSpacing(16)
        lay.addWidget(h1(tr("report_title")))
        lay.addWidget(muted(tr("report_text")))
        c = Card()
        self.cur = wrap("", 11.5, True)
        c.lay.addWidget(self.cur)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("report_lang")))
        self.rlang = QComboBox()
        self.rlang.addItem("العربية", "ar")
        self.rlang.addItem("English", "en")
        self.rlang.setCurrentIndex(0 if lang() == "ar" else 1)
        row.addWidget(self.rlang)
        row.addStretch()
        c.lay.addLayout(row)
        btns = QHBoxLayout()
        self.b_pdf = button(tr("export_pdf"), "primary", self.export_pdf)
        self.b_find = button(tr("export_csv_findings"), "", self.export_findings)
        self.b_data = button(tr("export_csv_data"), "", self.export_data)
        for b in (self.b_pdf, self.b_find, self.b_data):
            btns.addWidget(b)
        btns.addStretch()
        c.lay.addLayout(btns)
        self.last = muted("")
        c.lay.addWidget(self.last)
        lay.addWidget(c)
        inc = Card()
        inc.lay.addWidget(wrap(pick("التقرير فيه:", "The report contains:"), 11, True))
        for ar, en in (("النتيجة العامة وملخص الرحلة", "Overall verdict and trip summary"),
                       ("كل ملاحظة بالشرح: حصل إيه، الطبيعي، المعنى، تعمل إيه", "Each finding explained: what, normal, meaning, action"),
                       ("رسومات للقراءات مع تظليل الأماكن اللي فيها ملاحظات", "Charts with the findings shaded"),
                       ("جودة الداتا ومصدرها", "Data quality and source"),
                       ("الموديلات المستخدمة وحدود البرنامج", "Models used and limitations")):
            inc.lay.addWidget(wrap("✓ " + pick(ar, en), 10.5))
        lay.addWidget(inc)
        lay.addStretch()
        state.session_changed.connect(self.refresh)
        state.analysis_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        s = self.state.session
        on = s is not None
        self.b_pdf.setEnabled(on)
        self.b_data.setEnabled(on)
        self.b_find.setEnabled(on and self.state.result is not None)
        self.cur.setText(display_name(s.meta.source_name, lang()) if on else tr("no_trip"))

    def _ask(self, default: str, filt: str) -> str | None:
        stem = Path(self.state.session.meta.source_name.replace(":", "_")).stem
        path, _ = QFileDialog.getSaveFileName(self, tr("report_title"), f"{stem}_{default}", filt)
        return path or None

    def _done(self, path) -> None:
        self.state.busy_changed.emit(False, "")
        self.last.setText(f"{tr('saved_to')} {path}")
        if QMessageBox.question(self, tr("report_title"), f"{tr('saved_to')}\n{path}\n\n{tr('open_now')}") \
                == QMessageBox.Yes:
            os.startfile(str(path))  # noqa: S606 - a local file the user just created

    def export_pdf(self) -> None:
        path = self._ask("report.pdf", "PDF (*.pdf)")
        if not path:
            return
        st = self.state
        self.state.busy_changed.emit(True, "PDF…")
        w = Worker(build_pdf_report, path, st.session, st.quality, st.result, st.session_id,
                   st.vehicle_profile(), self.rlang.currentData())
        w.signals.finished.connect(self._done)
        w.signals.failed.connect(self.state._load_failed)
        self.state.pool.start(w)

    def export_findings(self) -> None:
        path = self._ask("findings.csv", "CSV (*.csv)")
        if path:
            df = findings_frame(self.state.findings, self.rlang.currentData())
            df.insert(0, "data_origin", self.state.session.meta.origin.value)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            self._done(path)

    def export_data(self) -> None:
        path = self._ask("data.csv", "CSV (*.csv)")
        if path:
            self._done(export_session_csv(self.state.session, path))
