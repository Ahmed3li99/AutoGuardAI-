"""Trip: replay with big value cards and charts; findings shaded on the charts."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QSlider, QVBoxLayout,
                               QWidget)

from ai.explain import display_name, mmss
from app import icons, theme
from app.i18n import lang, pick, tr
from app.state import AppState
from app.widgets.common import Card, Pill, ValueCard, button, muted, signal_label
from core.playback.replay import ALLOWED_SPEEDS
from core.schema import CANONICAL_FIELDS, PRIMARY_SIGNALS, TIME_COL, TimeBasis

MAX_POINTS = 3600
CARD_SIGNALS = ["speed_kmh", "rpm", "coolant_temp_c", "battery_voltage"]
DEFAULT_PLOTS = ["speed_kmh", "rpm", "coolant_temp_c", "battery_voltage"]


class TripPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 18, 28, 18)
        lay.setSpacing(12)

        top = QHBoxLayout()
        self.title = QLabel(tr("no_trip"))
        self.title.setObjectName("h2")
        top.addWidget(self.title, 1)
        self.origin = Pill("")
        top.addWidget(self.origin)
        lay.addLayout(top)

        cards = QGridLayout()
        cards.setSpacing(12)
        self.cards = {s: ValueCard(s) for s in CARD_SIGNALS}
        for i, s in enumerate(CARD_SIGNALS):
            cards.addWidget(self.cards[s], 0, i)
        lay.addLayout(cards)

        ctl = Card(margins=12)
        row = QHBoxLayout()
        self.b_play = button(tr("play"), "primary", self._play)
        self.b_pause = button(tr("pause"), "", state.pause)
        self.b_stop = button(tr("stop"), "", state.stop_replay)
        self.b_play.setIcon(icons.icon("play", "#FFFFFF", size=16))
        self.b_pause.setIcon(icons.icon("pause", theme.TEXT, size=16))
        self.b_stop.setIcon(icons.icon("stop", theme.TEXT, size=16))
        self.speed = QComboBox()
        for s in ALLOWED_SPEEDS:
            self.speed.addItem(f"{s:g}×", s)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(lambda _: state.set_speed(self.speed.currentData()))
        for w in (self.b_play, self.b_pause, self.b_stop, QLabel(tr("speed")), self.speed):
            row.addWidget(w)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self._seek)
        row.addWidget(self.slider, 1)
        self.time_lbl = QLabel("00:00 / 00:00")
        self.time_lbl.setStyleSheet("font-size:12pt; font-weight:600; background:transparent")
        row.addWidget(self.time_lbl)
        ctl.lay.addLayout(row)
        self.note = muted(tr("replay_note"))
        ctl.lay.addWidget(self.note)
        lay.addWidget(ctl)

        chart = Card(margins=10)
        sel = QHBoxLayout()
        sel.addWidget(muted(tr("signals_to_show") + ":"))
        self.checks: dict[str, QCheckBox] = {}
        for s in PRIMARY_SIGNALS:
            cb = QCheckBox(signal_label(s))
            cb.setChecked(s in DEFAULT_PLOTS)
            cb.toggled.connect(lambda _=False: self._rebuild_plots())
            self.checks[s] = cb
            sel.addWidget(cb)
        sel.addStretch()
        chart.lay.addLayout(sel)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(theme.CARD)
        chart.lay.addWidget(self.glw, 1)
        lay.addWidget(chart, 1)

        self.curves: dict[str, pg.PlotDataItem] = {}
        self.plots: list[pg.PlotItem] = []
        self._focus: tuple[float, float] | None = None
        state.session_changed.connect(self._on_session)
        state.analysis_changed.connect(self._rebuild_plots)
        state.replay_tick.connect(self._on_tick)
        state.replay_state_changed.connect(self._on_state)
        state.focus_time.connect(self._on_focus)
        self._on_session()

    # ---- session ----------------------------------------------------------
    def _on_session(self) -> None:
        s = self.state.session
        self._focus = None
        if s is None:
            self.title.setText(tr("no_trip"))
            self.origin.set("", theme.MUTED, "transparent")
            self._rebuild_plots()
            return
        self.title.setText(display_name(s.meta.source_name, lang()))
        o = s.meta.origin.value
        soft = {"recorded": theme.PRIMARY_SOFT, "synthetic": theme.SYNTHETIC_SOFT, "live_obd": theme.OK_SOFT}[o]
        self.origin.set(tr("origin_" + o), theme.ORIGIN_COLORS[o], soft)
        tb = {TimeBasis.RELATIVE_RECORDED: "time_basis_relative", TimeBasis.SYNTHETIC_CLOCK: "time_basis_synthetic",
              TimeBasis.ABSOLUTE: "time_basis_absolute"}[s.meta.time_basis]
        self.note.setText(tr("replay_note") + "  ·  " + pick("محور الوقت: ", "Time axis: ") + tr(tb))
        for sig, cb in self.checks.items():
            cb.setEnabled(sig in s.signals)
            if sig not in s.signals:
                cb.setChecked(False)
        self.state.set_speed(self.speed.currentData())
        self._rebuild_plots()

    def _selected(self) -> list[str]:
        s = self.state.session
        return [sig for sig, cb in self.checks.items() if cb.isChecked() and s and sig in s.signals][:5]

    def _rebuild_plots(self) -> None:
        self.glw.clear()
        self.curves.clear()
        self.plots = []
        s = self.state.session
        if s is None:
            return
        first = None
        for k, sig in enumerate(self._selected()):
            p = self.glw.addPlot(row=k, col=0)
            p.setLabel("left", f"{signal_label(sig)} [{CANONICAL_FIELDS[sig].unit}]")
            p.getAxis("left").enableAutoSIPrefix(False)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setMenuEnabled(False)
            p.getAxis("bottom").setStyle(showValues=k == len(self._selected()) - 1)
            if first is None:
                first = p
            else:
                p.setXLink(first)
            col = theme.PLOT_COLORS[k % len(theme.PLOT_COLORS)]
            self.curves[sig] = p.plot(pen=pg.mkPen(col, width=1.8))
            for f in self.state.findings:
                if f.confidence == "low":
                    continue
                reg = pg.LinearRegionItem((f.t_start_s, f.t_end_s + 1), movable=False,
                                          brush=pg.mkBrush(theme.SEVERITY_COLORS[f.severity] + "33"),
                                          pen=pg.mkPen(None))
                reg.setZValue(-10)
                p.addItem(reg)
            self.plots.append(p)
        if self.plots:
            self.plots[-1].setLabel("bottom", pick("الوقت (ثانية)", "time (s)"))
        self._on_tick()
        if self._focus and first is not None:
            first.setXRange(*self._focus, padding=0)

    # ---- replay -------------------------------------------------------------
    def _play(self) -> None:
        self._focus = None
        self.state.play()

    def _seek(self, v: int) -> None:
        e = self.state.engine
        if e:
            self._focus = None
            self.state.seek(e.duration_s * v / 1000.0)

    def _on_state(self, st: str) -> None:
        self.b_play.setText(tr("resume") if st == "paused" else tr("play"))

    def _on_focus(self, t0: float, t1: float) -> None:
        e = self.state.engine
        if e is None:
            return
        self.state.pause()
        lo, hi = max(e.start_s, t0 - 120), t1 + 120
        self.state.seek(min(hi, e.start_s + e.duration_s) - e.start_s)
        self._focus = (lo, hi)
        self._rebuild_plots()

    def _on_tick(self) -> None:
        e, s = self.state.engine, self.state.session
        if e is None or s is None:
            for c in self.cards.values():
                c.update_value(None, available=False)
            self.time_lbl.setText("00:00 / 00:00")
            return
        hist = e.history(MAX_POINTS)
        t = hist[TIME_COL].to_numpy(dtype=float)
        for sig, curve in self.curves.items():
            curve.setData(t, hist[sig].to_numpy(dtype=float) if sig in hist else np.array([]), connect="finite")
        row = e.current_row()
        active = {f.signal: f.severity for f in self.state.findings
                  if f.confidence != "low" and row is not None and f.t_start_s <= row[TIME_COL] <= f.t_end_s + 1}
        for sig, card in self.cards.items():
            avail = sig in s.signals
            val = None if row is None or not avail else float(row[sig])
            card.update_value(val, avail, theme.SEVERITY_COLORS.get(active.get(sig)))
        self.slider.blockSignals(True)
        self.slider.setValue(int(1000 * e.elapsed_s / e.duration_s) if e.duration_s else 0)
        self.slider.blockSignals(False)
        self.time_lbl.setText(f"{mmss(e.elapsed_s)} / {mmss(e.duration_s)}")
