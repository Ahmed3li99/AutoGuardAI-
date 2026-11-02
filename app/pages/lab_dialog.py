"""Scenario lab: try artificial faults on a copy of the current trip, and 'what if?'."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QSpinBox,
                               QTabWidget, QVBoxLayout, QWidget)

from ai.explain import NAMES, mmss
from ai.scenarios import DETECTORS, SCENARIO_KINDS, run_scenario, what_if_normal
from app import theme
from app.i18n import lang, pick
from app.state import AppState, Worker
from app.widgets.common import Card, button, h2, muted, signal_label, wrap
from core.analytics.injection import InjectedEvent
from core.schema import TIME_COL

DET_NAMES = {"rules": ("قواعد الحدود", "Limit rules"), "consistency": ("اتساق الحساسات", "Sensor consistency"),
             "expected_vs_actual": ("المتوقع مقابل الفعلي", "Expected vs actual"),
             "isolation_forest": ("Isolation Forest", "Isolation Forest"), "moment": ("MOMENT", "MOMENT")}
DEFAULT_MAG = {"voltage_decline": 11.6, "step_jump": 15.0}


class LabDialog(QDialog):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle(pick("معمل السيناريوهات", "Scenario lab"))
        self.resize(1180, 860)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        head = Card()
        head.lay.addWidget(h2(pick("معمل السيناريوهات — جرّب على رحلة حقيقية", "Scenario lab — experiment on a real trip")))
        head.lay.addWidget(wrap(pick(
            "البرنامج بياخد نسخة من الرحلة الحالية، يحط فيها اللي تختاره، ويحللها بنفس الطرق — وتشوف مين اكتشف إيه وإمتى. "
            "النسخة المعدّلة تجريبية ومابتتحفظش في السجل.",
            "A copy of the current trip is modified as you choose and analysed with the same methods, so you see which "
            "method found what and when. The modified copy is experimental and never saved to history."), 10.5,
            False, theme.MUTED))
        lay.addWidget(head)
        tabs = QTabWidget()
        tabs.addTab(self._build_fault_tab(), pick("جرّب عطل", "Try a fault"))
        tabs.addTab(self._build_whatif_tab(), pick("ماذا لو؟", "What if?"))
        lay.addWidget(tabs, 1)
        bb = QHBoxLayout()
        bb.addStretch()
        bb.addWidget(button(pick("قفل", "Close"), "", self.accept))
        lay.addLayout(bb)

    # ======================================================== try a fault
    def _build_fault_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        s = self.state.session
        g = QGridLayout()
        self.kind = QComboBox()
        for k, (ar, en, _, _) in SCENARIO_KINDS.items():
            self.kind.addItem(pick(ar, en), k)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.sig = QComboBox()
        self.start = QSpinBox()
        self.start.setSuffix(pick(" ث", " s"))
        self.dur = QSpinBox()
        self.dur.setRange(10, 3600)
        self.dur.setValue(180)
        self.dur.setSuffix(pick(" ث", " s"))
        self.mag = QDoubleSpinBox()
        self.mag.setRange(-500, 500)
        self.mag_lbl = QLabel(pick("المقدار", "Magnitude"))
        if s is not None:
            t0, t1 = float(s.data[TIME_COL].iloc[0]), float(s.data[TIME_COL].iloc[-1])
            self.start.setRange(int(t0), max(int(t0), int(t1) - 30))
            self.start.setValue(int(t0 + (t1 - t0) * 0.4))
        for i, (lab, wid) in enumerate(((pick("نوع التجربة", "Scenario"), self.kind), (pick("الحساس", "Sensor"), self.sig),
                                        (pick("بداية من", "Starts at"), self.start), (pick("المدة", "Duration"), self.dur))):
            g.addWidget(QLabel(lab), i // 2, (i % 2) * 2)
            g.addWidget(wid, i // 2, (i % 2) * 2 + 1)
        g.addWidget(self.mag_lbl, 2, 0)
        g.addWidget(self.mag, 2, 1)
        self.b_run = button(pick("شغّل التجربة", "Run experiment"), "primary", self.run_fault)
        g.addWidget(self.b_run, 2, 3)
        lay.addLayout(g)
        self.fault_status = muted("")
        lay.addWidget(self.fault_status)
        self.det_grid = QGridLayout()
        lay.addLayout(self.det_grid)
        self.fault_plot = pg.PlotWidget()
        self.fault_plot.setBackground(theme.CARD)
        self.fault_plot.showGrid(x=True, y=True, alpha=0.15)
        self.fault_plot.getAxis("left").enableAutoSIPrefix(False)
        lay.addWidget(self.fault_plot, 1)
        self.fault_diff = wrap("", 10.5)
        lay.addWidget(self.fault_diff)
        self._kind_changed()
        if s is None:
            self.b_run.setEnabled(False)
            self.fault_status.setText(pick("افتح رحلة الأول.", "Open a trip first."))
        return w

    def _kind_changed(self) -> None:
        k = self.kind.currentData()
        _, _, allowed, takes_mag = SCENARIO_KINDS[k]
        s = self.state.session
        avail = s.signals if s else []
        self.sig.clear()
        for sig in (allowed or [x for x in ("rpm", "speed_kmh", "coolant_temp_c", "battery_voltage", "engine_load_pct",
                                            "throttle_pct", "intake_air_temp_c", "manifold_pressure_kpa") if x in avail]):
            if sig in avail:
                self.sig.addItem(signal_label(sig), sig)
        self.mag.setVisible(takes_mag)
        self.mag_lbl.setVisible(takes_mag)
        if takes_mag:
            self.mag.setValue(DEFAULT_MAG.get(k, 0.0))
            self.mag_lbl.setText(pick("الفولت في الآخر", "Final voltage") if k == "voltage_decline"
                                 else pick("مقدار القفزة", "Jump size"))
        if hasattr(self, "b_run"):
            self.b_run.setEnabled(self.sig.count() > 0 and s is not None)

    def run_fault(self) -> None:
        k, sig = self.kind.currentData(), self.sig.currentData()
        ev = InjectedEvent(k, sig, float(self.start.value()), float(self.start.value() + self.dur.value()),
                           "scenario lab", self.mag.value() if SCENARIO_KINDS[k][3] else None)
        self.b_run.setEnabled(False)
        self.fault_status.setText(pick("بيحلّل النسخة المعدّلة…", "Analysing the modified copy…"))
        st = self.state
        w = Worker(run_scenario, st.session, [ev], st._registry_for_current(), st.rule_config(), st.result)
        w.signals.finished.connect(self._fault_done)
        w.signals.failed.connect(self._failed)
        st.pool.start(w)

    def _fault_done(self, r) -> None:
        self.b_run.setEnabled(True)
        ev = r.events[0]
        self.fault_status.setText(pick(
            f"العطل المصطنع من {mmss(ev.start_s)} لـ {mmss(ev.end_s)} على «{signal_label(ev.signal)}». مين لقاه؟",
            f"Artificial fault from {mmss(ev.start_s)} to {mmss(ev.end_s)} on '{signal_label(ev.signal)}'. Who found it?"))
        while self.det_grid.count():
            it = self.det_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        caught = {d: v for d, v in r.per_detector.items() if v["caught"] and v["caught"][0]}
        fastest = min(caught, key=lambda d: caught[d]["delay_s"]) if caught else None
        for i, det in enumerate(DETECTORS):
            v = r.per_detector[det]
            name = QLabel(pick(*DET_NAMES[det]))
            name.setStyleSheet("font-weight:600")
            if not v["ran"]:
                txt, col = pick("ماشتغلش", "did not run"), theme.MUTED
            elif v["caught"][0]:
                txt = pick(f"✓ لقاه بعد {v['delay_s']:.0f} ثانية", f"✓ found it after {v['delay_s']:.0f} s")
                txt += pick("  ★ الأسرع", "  ★ fastest") if det == fastest else ""
                col = theme.OK
            else:
                txt, col = pick("✗ مالقاهوش", "✗ missed it"), theme.WARNING
            res = QLabel(txt)
            res.setStyleSheet(f"color:{col}; font-weight:700")
            self.det_grid.addWidget(name, i // 3, (i % 3) * 2)
            self.det_grid.addWidget(res, i // 3, (i % 3) * 2 + 1)
        self._plot(self.fault_plot, r, ev.signal, ev.start_s, ev.end_s, pick("بعد التجربة", "with the fault"))
        new = [f for f in r.diff["new"] if f.confidence != "low"]
        self.fault_diff.setText(pick("ملاحظات جديدة ظهرت: ", "New findings: ") + (
            pick("، ", ", ").join(f.text(lang())["title"] for f in new) or pick("ولا حاجة", "none")))

    # ======================================================== what if
    def _build_whatif_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(wrap(pick(
            "اختار ملاحظة من الرحلة، والبرنامج يعيد التحليل كأن الحساس ده فضل طبيعي (بالقيمة اللي الموديل متوقعها) "
            "في الفترة دي. تشوف أنهي ملاحظات بتختفي وأنهي بتفضل. دي تجربة على الموديل، مش إثبات إن تصليح القطعة "
            "هيحل المشكلة.",
            "Pick a finding; the trip is re-analysed as if that sensor had stayed normal (at the value the model "
            "expects) during that time. See which findings disappear and which remain. This is an experiment on "
            "the model, not proof that repairing a part would fix anything."), 10.5, False, theme.MUTED))
        row = QHBoxLayout()
        self.wf = QComboBox()
        nbm = (self.state._registry_for_current().expected_vs_actual(self.state.session.signals)
               if self.state.session is not None else None)
        targets = set(nbm.models) if nbm is not None else set()
        for f in self.state.findings:
            if f.signal in targets and f.confidence != "low":
                self.wf.addItem(f"{f.text(lang())['title']}  ({mmss(f.t_start_s)}–{mmss(f.t_end_s)})", f)
        row.addWidget(self.wf, 1)
        self.b_wf = button(pick("لو كان طبيعي؟", "What if it were normal?"), "primary", self.run_whatif)
        self.b_wf.setEnabled(self.wf.count() > 0)
        row.addWidget(self.b_wf)
        lay.addLayout(row)
        self.wf_status = muted("" if self.wf.count() else pick(
            "مفيش ملاحظات على حساس الموديل يقدر يتوقعه في الرحلة دي — حلّل رحلة فيها ملاحظات (زي العرض التجريبي).",
            "No findings on a sensor the model can predict in this trip — analyse a trip with findings (e.g. the demo)."))
        lay.addWidget(self.wf_status)
        self.wf_result = wrap("", 11)
        lay.addWidget(self.wf_result)
        self.wf_plot = pg.PlotWidget()
        self.wf_plot.setBackground(theme.CARD)
        self.wf_plot.showGrid(x=True, y=True, alpha=0.15)
        self.wf_plot.getAxis("left").enableAutoSIPrefix(False)
        lay.addWidget(self.wf_plot, 1)
        return w

    def run_whatif(self) -> None:
        f = self.wf.currentData()
        self.b_wf.setEnabled(False)
        self.wf_status.setText(pick("بيعيد التحليل…", "Re-analysing…"))
        st = self.state
        self._wf_finding = f
        w = Worker(what_if_normal, st.session, f.signal, f.t_start_s - 5, f.t_end_s + 5, st._registry_for_current(),
                   st.rule_config(), st.result)
        w.signals.finished.connect(self._whatif_done)
        w.signals.failed.connect(self._failed)
        st.pool.start(w)

    def _whatif_done(self, r) -> None:
        self.b_wf.setEnabled(True)
        f = self._wf_finding
        gone = [g for g in r.diff["gone"] if g.confidence != "low"]
        kept = [g for g in r.diff["kept"] if g.confidence != "low"]
        name = signal_label(f.signal)
        self.wf_status.setText(pick(f"لو «{name}» فضل طبيعي من {mmss(f.t_start_s)} لـ {mmss(f.t_end_s)}:",
                                    f"If '{name}' had stayed normal from {mmss(f.t_start_s)} to {mmss(f.t_end_s)}:"))
        L = lang()
        self.wf_result.setText(
            "<span style='color:%s'><b>%s</b> %s</span><br><span style='color:%s'><b>%s</b> %s</span>" % (
                theme.OK, pick("✓ هتختفي:", "✓ would disappear:"),
                pick("، ", ", ").join(g.text(L)["title"] for g in gone) or pick("ولا حاجة", "nothing"),
                theme.ATTENTION, pick("● هتفضل:", "● would remain:"),
                pick("، ", ", ").join(g.text(L)["title"] for g in kept) or pick("ولا حاجة", "nothing")))
        self._plot(self.wf_plot, r, f.signal, f.t_start_s, f.t_end_s, pick("لو كان طبيعي", "if normal"))

    # ======================================================== shared
    def _plot(self, plot: pg.PlotWidget, r, sig: str, t0: float, t1: float, mod_label: str) -> None:
        plot.clear()
        plot.addLegend(offset=(-10, -10))      # bottom corner, away from the lines
        lo, hi = t0 - 180, t1 + 180
        for df, col, lab, width in ((r.original.data, theme.MUTED, pick("الأصلي", "original"), 2.4),
                                    (r.modified.data, theme.PRIMARY, mod_label, 1.6)):
            seg = df[(df[TIME_COL] >= lo) & (df[TIME_COL] <= hi)]
            plot.plot(seg[TIME_COL].to_numpy(float), seg[sig].to_numpy(float), pen=pg.mkPen(col, width=width),
                      name=lab, connect="finite")
        reg = pg.LinearRegionItem((t0, t1), movable=False, brush=pg.mkBrush(theme.SYNTHETIC + "22"), pen=pg.mkPen(None))
        reg.setZValue(-10)
        plot.addItem(reg)
        caught = [(d, v) for d, v in r.per_detector.items() if v["caught"] and v["caught"][0]]
        for k, (d, v) in enumerate(sorted(caught, key=lambda x: x[1]["delay_s"])):
            # stagger label heights so methods that fired close together stay readable
            line = pg.InfiniteLine(pos=t0 + v["delay_s"], angle=90, pen=pg.mkPen(theme.OK, style=Qt.DashLine),
                                   label=pick(*DET_NAMES[d]),
                                   labelOpts={"position": 0.92 - 0.12 * k, "color": theme.OK, "fill": theme.CARD})
            plot.addItem(line)
        ar, en, unit = NAMES.get(sig, (sig, sig, ""))
        plot.setLabel("left", f"{pick(ar, en)} [{unit}]")
        plot.setLabel("bottom", pick("الوقت (ثانية)", "time (s)"))

    def _failed(self, msg: str) -> None:
        self.b_run.setEnabled(True)
        self.b_wf.setEnabled(self.wf.count() > 0)
        self.fault_status.setText("✖ " + msg)
        self.wf_status.setText("✖ " + msg)

