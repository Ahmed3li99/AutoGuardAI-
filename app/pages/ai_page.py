"""AI & Anomalies: detector status, scores vs threshold, alert evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QTextEdit,
                               QVBoxLayout, QWidget)

from ai.classifier_optional import UNAVAILABLE_MESSAGE
from app import theme
from app.state import AppState
from app.widgets.common import fill_table, h1, muted, readonly_table
from core.schema import TIME_COL

EVIDENCE_MARGIN_S = 90


class AIPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(h1("AI & Anomalies"))
        top.addStretch()
        self.b_run = QPushButton("Run analysis")
        self.b_run.setObjectName("primary")
        self.b_run.clicked.connect(state.run_analysis_async)
        top.addWidget(self.b_run)
        lay.addLayout(top)
        lay.addWidget(muted(
            "Detectors: heuristic rules, an Isolation Forest baseline and the pretrained MOMENT-1-small model "
            "(zero-shot reconstruction error). Anomaly scores mark statistically unusual patterns relative to "
            "calibration trips — they are not diagnoses. " + UNAVAILABLE_MESSAGE + "."))
        self.status_tbl = readonly_table()
        self.status_tbl.setMaximumHeight(150)
        lay.addWidget(self.status_tbl)

        split = QSplitter(Qt.Vertical)
        self.score_plot = pg.GraphicsLayoutWidget()
        split.addWidget(self.score_plot)
        bottom = QSplitter(Qt.Horizontal)
        abox = QGroupBox("Alerts (click one for evidence)")
        al = QVBoxLayout(abox)
        self.alert_tbl = readonly_table()
        self.alert_tbl.itemSelectionChanged.connect(self._show_evidence)
        al.addWidget(self.alert_tbl)
        bottom.addWidget(abox)
        ebox = QGroupBox("Evidence window")
        el = QVBoxLayout(ebox)
        self.ev_plot = pg.GraphicsLayoutWidget()
        self.ev_text = QTextEdit()
        self.ev_text.setReadOnly(True)
        self.ev_text.setMaximumHeight(120)
        el.addWidget(self.ev_plot, 1)
        el.addWidget(self.ev_text)
        bottom.addWidget(ebox)
        split.addWidget(bottom)
        split.setSizes([260, 420])
        lay.addWidget(split, 1)
        self.busy = QLabel("")
        lay.addWidget(self.busy)

        state.analysis_changed.connect(self.refresh)
        state.session_changed.connect(self.refresh)
        state.busy_changed.connect(self._busy)
        self.refresh()

    def _busy(self, on: bool, msg: str) -> None:
        self.b_run.setEnabled(not on)
        self.busy.setText(msg)

    def refresh(self) -> None:
        res, s = self.state.result, self.state.session
        self.b_run.setEnabled(s is not None)
        self.score_plot.clear()
        self.ev_plot.clear()
        self.ev_text.clear()
        if res is None:
            fill_table(self.status_tbl, pd.DataFrame({"info": ["No analysis yet — click 'Run analysis'."]}))
            fill_table(self.alert_tbl, pd.DataFrame())
            return
        rows = []
        for det, st in res.detector_status.items():
            rows.append({"detector": det, "status": st.get("status"), "kind": st.get("kind", ""),
                         "version": st.get("version", ""), "device": st.get("device", ""),
                         "calibration": st.get("calibration", ""),
                         "run time [s]": round(res.timing_s[det], 3) if det in res.timing_s else None,
                         "reason / note": st.get("reason") or st.get("note") or st.get("warning") or ""})
        rows.append({"detector": "fault_classifier", "status": "unavailable", "kind": "fault_classifier",
                     "reason / note": UNAVAILABLE_MESSAGE})
        fill_table(self.status_tbl, pd.DataFrame(rows))

        first = None
        for k, (det, sc) in enumerate(res.scores.items()):
            p = self.score_plot.addPlot(row=k, col=0)
            p.setLabel("left", det)
            p.getAxis("left").enableAutoSIPrefix(False)   # show raw score values, not x0.001
            p.showGrid(x=True, y=True, alpha=0.15)
            y = np.array([np.nan if v is None else v for v in sc.scores], dtype=float)
            p.plot(np.asarray(sc.t_s, float), y, pen=pg.mkPen(theme.PLOT_COLORS[k + 1], width=1), connect="finite")
            if sc.threshold:
                p.addLine(y=sc.threshold["attention"], pen=pg.mkPen(theme.ATTENTION, style=Qt.DashLine))
                p.addLine(y=sc.threshold["warning"], pen=pg.mkPen(theme.WARNING, style=Qt.DashLine))
            if first is None:
                first = p
            else:
                p.setXLink(first)
        if first is None:
            self.score_plot.addLabel("No ML scores (models unavailable — see status table). Rules still ran.")

        self._alerts = res.alerts
        fill_table(self.alert_tbl, pd.DataFrame([{
            "start [s]": a.t_start_s, "end [s]": a.t_end_s, "severity": a.severity, "detector": a.detector,
            "signals": ", ".join(a.signals), "peak": a.peak_score, "threshold": a.threshold,
            "origin": a.origin} for a in res.alerts]), float_fmt="{:.4g}")
        for i, a in enumerate(res.alerts):
            item = self.alert_tbl.item(i, 2)
            if item:
                item.setForeground(pg.mkColor(theme.SEVERITY_COLORS[a.severity]))

    def _show_evidence(self) -> None:
        rows = self.alert_tbl.selectionModel().selectedRows()
        s = self.state.session
        if not rows or s is None or not getattr(self, "_alerts", None):
            return
        a = self._alerts[rows[0].row()]
        df = s.data
        lo, hi = a.t_start_s - EVIDENCE_MARGIN_S, a.t_end_s + EVIDENCE_MARGIN_S
        w = df[(df[TIME_COL] >= lo) & (df[TIME_COL] <= hi)]
        self.ev_plot.clear()
        first = None
        for k, sig in enumerate([x for x in a.signals if x in w][:3]):
            p = self.ev_plot.addPlot(row=k, col=0)
            p.setLabel("left", sig)
            p.plot(w[TIME_COL].to_numpy(float), w[sig].to_numpy(float),
                   pen=pg.mkPen(theme.PLOT_COLORS[k], width=1.3), connect="finite")
            reg = pg.LinearRegionItem((a.t_start_s, a.t_end_s + 1), movable=False,
                                      brush=pg.mkBrush(theme.SEVERITY_COLORS[a.severity] + "40"))
            p.addItem(reg)
            if first is None:
                first = p
            else:
                p.setXLink(first)
        vals = ", ".join(f"{k}={v}" for k, v in a.values.items())
        self.ev_text.setHtml(f"<b>{a.severity.upper()}</b> · {a.detector} ({a.detector_kind}) · model "
                             f"{a.model_version} · origin: {a.origin}<br>{a.explanation}<br>"
                             f"<span style='color:{theme.MUTED}'>Values at peak: {vals}</span>")
        if self.state.engine:
            self.state.seek(max(0.0, a.t_start_s - self.state.engine.start_s))
