"""Overview / dashboard with signal cards, status badges and recent alerts."""
from __future__ import annotations

from PySide6.QtWidgets import (QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QPushButton,
                               QVBoxLayout, QWidget)

from ai.pipeline import overall_status
from app import theme
from app.state import AppState
from app.widgets.common import Badge, SignalCard, h1, muted
from core.schema import PRIMARY_SIGNALS


class DashboardPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(h1("Overview"))
        top.addStretch()
        self.origin_badge = Badge("No session", theme.MUTED)
        self.status_badge = Badge("—", theme.MUTED)
        top.addWidget(QLabel("Source:"))
        top.addWidget(self.origin_badge)
        top.addWidget(QLabel("Status:"))
        top.addWidget(self.status_badge)
        lay.addLayout(top)
        self.src_lbl = muted("Import a file or load a sample/synthetic demo from 'Data Import & Quality'.")
        lay.addWidget(self.src_lbl)
        self.time_lbl = QLabel("")
        lay.addWidget(self.time_lbl)

        grid = QGridLayout()
        self.cards = {f: SignalCard(f) for f in PRIMARY_SIGNALS}
        for i, f in enumerate(PRIMARY_SIGNALS):
            grid.addWidget(self.cards[f], i // 4, i % 4)
        lay.addLayout(grid)

        row = QHBoxLayout()
        sbox = QGroupBox("Signal status (from rules and anomaly alerts so far)")
        self.sgrid = QGridLayout(sbox)
        row.addWidget(sbox, 1)
        abox = QGroupBox("Recent alerts (up to replay position)")
        al = QVBoxLayout(abox)
        self.alerts = QListWidget()
        al.addWidget(self.alerts)
        b = QPushButton("Open AI && Anomalies")
        b.clicked.connect(lambda: self.window().navigate("AI & Anomalies"))
        al.addWidget(b)
        row.addWidget(abox, 1)
        lay.addLayout(row, 1)
        hbox = QGroupBox("Vehicle Health Index")
        hl = QVBoxLayout(hbox)
        hl.addWidget(muted("N/A — intentionally not computed. No dataset available to this project contains "
                           "validated mechanical-health ground truth, so a 0–100 % score would be invented. "
                           "Use the per-signal status above and the evidence on the AI page instead."))
        lay.addWidget(hbox)

        state.session_changed.connect(self.refresh)
        state.analysis_changed.connect(self.refresh)
        state.replay_tick.connect(self._tick)
        self.refresh()

    def refresh(self) -> None:
        s = self.state.session
        if s is None:
            for c in self.cards.values():
                c.update_value(None, available=False)
            return
        self.origin_badge.set(s.meta.origin.label, theme.ORIGIN_COLORS[s.meta.origin.value])
        st = overall_status(self.state.result, s)
        self.status_badge.set(st, next((c for k, c in theme.STATUS_COLORS.items() if st.startswith(k)), theme.MUTED))
        self.src_lbl.setText(f"{s.meta.source_name} · {s.meta.time_basis.label} · {len(s.data)} rows · "
                             f"{s.duration_s:,.0f} s" + (f" · dataset: {s.meta.dataset_id}" if s.meta.dataset_id else ""))
        while self.sgrid.count():
            w = self.sgrid.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._status_badges = {}
        for i, f in enumerate(PRIMARY_SIGNALS):
            self.sgrid.addWidget(QLabel(f), i, 0)
            b = Badge("", theme.MUTED)
            self._status_badges[f] = b
            self.sgrid.addWidget(b, i, 1)
        self._tick()

    def _tick(self) -> None:
        s, e = self.state.session, self.state.engine
        if s is None:
            return
        row = e.current_row() if e else None
        for f, card in self.cards.items():
            avail = f in s.signals
            card.update_value(None if row is None or not avail else float(row[f]), available=avail)
        if e:
            self.time_lbl.setText(f"Replay time: {e.position_s:,.0f} s ({e.state.value})")
        pos = e.position_s if e else float("inf")
        res = self.state.result
        past = [a for a in (res.alerts if res else []) if a.t_start_s <= pos]
        if self.alerts.count() != len(past):
            self.alerts.clear()
            for a in reversed(past[-50:]):
                self.alerts.addItem(f"[{a.severity.upper()}] {a.t_start_s:,.0f}-{a.t_end_s:,.0f} s  "
                                    f"{a.detector}  ({', '.join(a.signals)})")
        for f, b in getattr(self, "_status_badges", {}).items():
            if f not in s.signals:
                b.set("Insufficient Data", theme.MUTED)
            elif res is None:
                b.set("Not analysed", theme.MUTED)
            else:
                # ML anomalies alone never turn a signal red (see overall_status)
                sev = {a.severity if a.detector_kind == "heuristic_rule" else "attention"
                       for a in past if f in a.signals}
                if "warning" in sev:
                    b.set("Warning", theme.WARNING)
                elif "attention" in sev:
                    b.set("Attention", theme.ATTENTION)
                else:
                    b.set("Normal", theme.OK)
