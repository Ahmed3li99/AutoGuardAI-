"""Recorded replay / vehicle monitor."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QSlider, QSplitter, QVBoxLayout, QWidget)

from app import theme
from app.state import AppState
from app.widgets.common import h1, muted
from core.playback.replay import ALLOWED_SPEEDS
from core.schema import CANONICAL_FIELDS, PRIMARY_SIGNALS, TIME_COL

MAX_POINTS = 3600          # bounded plot buffer (rows)
MAX_PLOTS = 5


class ReplayPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(h1("Replay / Vehicle Monitor"))
        top.addStretch()
        self.mode = QLabel("")
        top.addWidget(self.mode)
        lay.addLayout(top)
        lay.addWidget(muted("Replays recorded (or synthetic) data at its recorded timing. This is NOT a live "
                            "connection to a vehicle."))

        ctl = QHBoxLayout()
        self.b_play = QPushButton("▶ Play")
        self.b_play.setObjectName("primary")
        self.b_pause = QPushButton("⏸ Pause")
        self.b_stop = QPushButton("⏹ Stop")
        self.b_restart = QPushButton("⟲ Restart")
        self.speed = QComboBox()
        for s in ALLOWED_SPEEDS:
            self.speed.addItem(f"{s:g}×", s)
        self.b_play.clicked.connect(self._play)
        self.b_pause.clicked.connect(state.pause)
        self.b_stop.clicked.connect(state.stop_replay)
        self.b_restart.clicked.connect(self._restart)
        self.speed.currentIndexChanged.connect(lambda _: state.set_speed(self.speed.currentData()))
        for w in (self.b_play, self.b_pause, self.b_stop, self.b_restart, QLabel("Speed:"), self.speed):
            ctl.addWidget(w)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self._seek)
        ctl.addWidget(self.slider, 1)
        self.time_lbl = QLabel("—")
        self.time_lbl.setMinimumWidth(260)
        ctl.addWidget(self.time_lbl)
        lay.addLayout(ctl)

        split = QSplitter(Qt.Horizontal)
        self.sig_list = QListWidget()
        self.sig_list.setMaximumWidth(220)
        self.sig_list.itemChanged.connect(lambda _: self._rebuild_plots())
        split.addWidget(self.sig_list)
        self.glw = pg.GraphicsLayoutWidget()
        split.addWidget(self.glw)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)
        self.curves: dict[str, pg.PlotDataItem] = {}
        self.plots: list[pg.PlotItem] = []
        self._regions_drawn: set[int] = set()

        state.session_changed.connect(self._on_session)
        state.replay_tick.connect(self._on_tick)
        state.replay_state_changed.connect(self._on_state)
        state.analysis_changed.connect(self._reset_regions)
        self._on_session()

    # ---- session --------------------------------------------------------
    def _on_session(self) -> None:
        s = self.state.session
        self.sig_list.blockSignals(True)
        self.sig_list.clear()
        if s is None:
            self.mode.setText("")
            self.sig_list.blockSignals(False)
            self._rebuild_plots()
            return
        color = theme.ORIGIN_COLORS[s.meta.origin.value]
        self.mode.setText(f"<span style='background:{color};color:white;padding:4px 10px;border-radius:8px'>"
                          f"{s.meta.origin.label}</span>  {s.meta.source_name}")
        default = [x for x in PRIMARY_SIGNALS if x in s.signals][:4]
        for sig in [x for x in PRIMARY_SIGNALS if x in s.signals] + [x for x in s.signals if x not in PRIMARY_SIGNALS]:
            it = QListWidgetItem(f"{sig} [{CANONICAL_FIELDS[sig].unit}]")
            it.setData(Qt.UserRole, sig)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if sig in default else Qt.Unchecked)
            self.sig_list.addItem(it)
        self.sig_list.blockSignals(False)
        self._rebuild_plots()

    def _selected(self) -> list[str]:
        out = []
        for i in range(self.sig_list.count()):
            it = self.sig_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out[:MAX_PLOTS]

    def _rebuild_plots(self) -> None:
        self.glw.clear()
        self.curves.clear()
        self.plots = []
        self._regions_drawn = set()
        s = self.state.session
        if s is None:
            return
        first = None
        for k, sig in enumerate(self._selected()):
            p = self.glw.addPlot(row=k, col=0)
            p.setLabel("left", sig, units=None)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setMenuEnabled(False)
            if first is None:
                first = p
            else:
                p.setXLink(first)
            self.curves[sig] = p.plot(pen=pg.mkPen(theme.PLOT_COLORS[k % len(theme.PLOT_COLORS)], width=1.2))
            self.plots.append(p)
        if self.plots:
            self.plots[-1].setLabel("bottom", f"time [s] — {s.meta.time_basis.label}")
        self._on_tick()

    def _reset_regions(self) -> None:
        self._rebuild_plots()

    # ---- replay ---------------------------------------------------------
    def _play(self) -> None:
        self.state.play()   # resumes from the playhead when paused

    def _restart(self) -> None:
        self.state.stop_replay()
        self.state.play()

    def _seek(self, v: int) -> None:
        e = self.state.engine
        if e:
            self.state.seek(e.duration_s * v / 1000.0)
            self._rebuild_plots()   # alert regions reflect the new playhead

    def _on_state(self, st: str) -> None:
        self.b_play.setText("▶ Resume" if st == "paused" else "▶ Play")

    def _on_tick(self) -> None:
        e = self.state.engine
        if e is None:
            self.time_lbl.setText("—")
            return
        hist = e.history(MAX_POINTS)
        t = hist[TIME_COL].to_numpy(dtype=float)
        for sig, curve in self.curves.items():
            y = hist[sig].to_numpy(dtype=float) if sig in hist else np.array([])
            curve.setData(t, y, connect="finite")
        self.slider.blockSignals(True)
        self.slider.setValue(int(1000 * e.elapsed_s / e.duration_s) if e.duration_s else 0)
        self.slider.blockSignals(False)
        self.time_lbl.setText(f"t = {e.position_s:,.0f} s  ({e.elapsed_s:,.0f} / {e.duration_s:,.0f} s)  "
                              f"[{e.state.value}]")
        res = self.state.result
        if res:
            for i, a in enumerate(res.alerts):
                if i in self._regions_drawn or a.t_start_s > e.position_s:
                    continue
                self._regions_drawn.add(i)
                col = theme.SEVERITY_COLORS[a.severity]
                for p in self.plots:
                    reg = pg.LinearRegionItem((a.t_start_s, a.t_end_s + 1), movable=False,
                                              brush=pg.mkBrush(col + "40"), pen=pg.mkPen(None))
                    reg.setZValue(-10)
                    p.addItem(reg)
