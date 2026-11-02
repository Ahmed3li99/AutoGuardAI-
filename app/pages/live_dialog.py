"""Live OBD-II connection (auto-detect adapter, identify car, live values, record & analyse)."""
from __future__ import annotations

import threading
import time

from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtWidgets import QCheckBox, QDialog, QGridLayout, QHBoxLayout, QVBoxLayout

from app import theme
from app.i18n import pick, tr
from app.state import AppState
from app.widgets.common import Card, Pill, ValueCard, button, h2, muted, signal_label, wrap
from core.data_sources.base import DataSourceError, Reading
from core.data_sources.obd_optional import OBDDataSource, load_obd_module
from core.validation.quality import build_quality_report

LIVE_CARDS = ["speed_kmh", "rpm", "coolant_temp_c", "battery_voltage", "engine_load_pct", "throttle_pct"]


class _Signals(QObject):
    connected = Signal(dict)
    reading = Signal(object)
    failed = Signal(str)
    stopped = Signal()


class _LiveWorker(QRunnable):
    """Connects (can take several seconds while python-OBD scans ports) and then
    polls about once per second until stopped. Runs off the GUI thread."""

    def __init__(self, source: OBDDataSource, period_s: float = 1.0):
        super().__init__()
        self.source, self.period = source, period_s
        self.sig = _Signals()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self.source.open()
            info = self.source.identify()
            info["signals"] = self.source.supported_signals()
            info["dtcs"] = self.source.read_dtcs()
            self.sig.connected.emit(info)
            while not self._stop.is_set():
                t0 = time.monotonic()
                self.sig.reading.emit(self.source.poll_once())
                self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))
        except DataSourceError as e:
            self.sig.failed.emit(str(e))
        except Exception as e:  # serial/driver errors are platform specific
            self.sig.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            self.source.close()
            self.sig.stopped.emit()


class _RecordedLive:
    """Wraps what was recorded so it can go through the normal load/analyse path."""

    def __init__(self, source: OBDDataSource):
        self.src = source
        self.quality = None

    def read_session(self):
        s = self.src.read_session()
        s.data = s.data.sort_values("t_s").reset_index(drop=True)
        self.quality = build_quality_report(s, len(s.data), {}, {}, 0, [])
        return s


class LiveDialog(QDialog):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle(tr("live_title"))
        self.resize(980, 720)
        self.worker: _LiveWorker | None = None
        self.source: OBDDataSource | None = None
        self.recording = False
        self._save_after_stop = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(12)

        intro = Card()
        intro.lay.addWidget(h2(tr("live_title")))
        intro.lay.addWidget(wrap(tr("live_steps"), 10.5))
        intro.lay.addWidget(wrap(tr("live_safety"), 10, True, theme.WARNING))
        intro.lay.addWidget(wrap(tr("live_untested"), 9.5, False, theme.MUTED))
        row = QHBoxLayout()
        self.b_connect = button(tr("connect"), "primary", lambda: self.start(simulated=False))
        self.b_sim = button(tr("simulate"), "", lambda: self.start(simulated=True))
        self.b_disc = button(tr("disconnect"), "", self.stop)
        self.b_disc.setEnabled(False)
        self.status = Pill(tr("not_connected"), theme.MUTED, "#EEF1F5")
        for w in (self.b_connect, self.b_sim, self.b_disc):
            row.addWidget(w)
        row.addStretch()
        row.addWidget(self.status)
        intro.lay.addLayout(row)
        self.active_check = QCheckBox(pick("الفحص النشط: ترتيب القراءات حسب المعلومة الناقصة (تجريبي)",
                                          "Active acquisition: information-guided sampling (experimental)"))
        intro.lay.addWidget(self.active_check)
        self.active_info = wrap("", 9.5)
        intro.lay.addWidget(self.active_info)
        lay.addWidget(intro)

        self.info = Card()
        self.info_lbl = wrap("", 10.5)
        self.info.lay.addWidget(self.info_lbl)
        self.info.hide()
        lay.addWidget(self.info)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.cards = {s: ValueCard(s) for s in LIVE_CARDS}
        for i, s in enumerate(LIVE_CARDS):
            grid.addWidget(self.cards[s], i // 3, i % 3)
        lay.addLayout(grid)

        rec = QHBoxLayout()
        self.b_rec = button(tr("record"), "primary", self.toggle_record)
        self.b_rec.setEnabled(False)
        self.rec_lbl = muted("")
        rec.addWidget(self.b_rec)
        rec.addWidget(self.rec_lbl, 1)
        rec.addWidget(button(tr("close"), "", self.close))
        lay.addLayout(rec)

    # ---- connection -------------------------------------------------------
    def start(self, simulated: bool) -> None:
        if simulated:
            from core.data_sources.obd_simulator import simulated_obd_module
            mod = simulated_obd_module(speedup=1.0)
        else:
            mod = load_obd_module()
            if mod is None:
                self._fail(tr("obd_missing"))
                return
        self.source = OBDDataSource(None, enabled=True, obd_module=mod, simulated=simulated,
                                    active=self.active_check.isChecked())
        self.active_check.setEnabled(False)
        self.worker = _LiveWorker(self.source)
        self.worker.sig.connected.connect(self._connected)
        self.worker.sig.reading.connect(self._reading)
        self.worker.sig.failed.connect(self._fail)
        self.worker.sig.stopped.connect(self._stopped)
        self.status.set(tr("searching"), theme.ATTENTION, theme.ATTENTION_SOFT)
        self.b_connect.setEnabled(False)
        self.b_sim.setEnabled(False)
        self.b_disc.setEnabled(True)
        self.state.pool.start(self.worker)

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()

    def _connected(self, info: dict) -> None:
        sim = self.source is not None and self.source.simulated
        self.status.set(tr("connected") + (pick(" (محاكاة)", " (simulated)") if sim else ""), theme.OK, theme.OK_SOFT)
        sigs = pick("، ", ", ").join(signal_label(s) for s in info["signals"])
        dtcs = info.get("dtcs") or []
        dtc_txt = (pick("أكواد أعطال متسجلة في العربية: ", "Trouble codes stored in the car: ")
                   + "; ".join(f"{c} {d}" for c, d in dtcs)) if dtcs else \
            pick("العربية مفيهاش أكواد أعطال متسجلة.", "The car reports no stored trouble codes.")
        lines = [pick("المنفذ: ", "Port: ") + str(info.get("port") or "-"),
                 pick("البروتوكول: ", "Protocol: ") + str(info.get("protocol") or "-"),
                 pick("رقم الشاسيه (VIN): ", "VIN: ") + (info.get("vin") or pick("العربية مابتبعتوش", "not reported by the car")),
                 tr("supported_signals") + " " + (sigs or "-"), dtc_txt]
        self.info_lbl.setText("\n".join(lines))
        self.info.show()
        for s, c in self.cards.items():
            c.update_value(None, s in info["signals"])
        self.b_rec.setEnabled(True)

    def _reading(self, r: Reading) -> None:
        for s, c in self.cards.items():
            if s in r.values:
                c.update_value(r.values[s])
        if self.recording and self.source:
            self.rec_lbl.setText(tr("recording", n=len(self.source.recorded)))
        if self.source and self.source.scheduler is not None:
            names = [signal_label(p["signal"]) for p in self.source.acquisition_summary.get("next_queries", [])]
            self.active_info.setText(pick("القراءات المختارة: ", "Selected readings: ") + ", ".join(names)
                                    + pick(" — القراءات الأخرى ليست حديثة في هذه الدورة.",
                                           " — other displayed readings are not fresh in this cycle."))

    def _fail(self, msg: str) -> None:
        self.status.set("✖ " + tr("error"), theme.WARNING, theme.WARNING_SOFT)
        self.info_lbl.setText(msg)
        self.info.show()
        self.b_connect.setEnabled(True)
        self.b_sim.setEnabled(True)

    def _stopped(self) -> None:
        if self.recording:
            self.recording = False
            self._save_after_stop = True  # Preserve partial recordings after a lost connection.
            self.b_rec.setText(tr("record"))
        self.active_check.setEnabled(True)
        self.b_connect.setEnabled(True)
        self.b_sim.setEnabled(True)
        self.b_disc.setEnabled(False)
        self.b_rec.setEnabled(False)
        if self.status.text().startswith(tr("connected")):
            self.status.set(tr("not_connected"), theme.MUTED, "#EEF1F5")
        if self._save_after_stop and self.source is not None:
            self._save_after_stop = False
            if len(self.source.recorded) < 5:
                self.rec_lbl.setText(pick("التسجيل قصير جداً.", "Recording too short."))
                return
            self.state.load_source_async(_RecordedLive(self.source))
            self.accept()
            if self.parent() is not None:
                self.parent().navigate("results")

    # ---- recording ---------------------------------------------------------
    def toggle_record(self) -> None:
        if not self.source:
            return
        if not self.recording:
            self.source.recorded.clear()
            self.recording = True
            self.b_rec.setText(tr("stop_record"))
            self.rec_lbl.setText(tr("recording", n=0))
            return
        self.recording = False
        self.b_rec.setText(tr("record"))
        self._save_after_stop = True
        self.b_rec.setEnabled(False)
        self.stop()

    def closeEvent(self, e):  # noqa: N802
        if self.recording:
            self.toggle_record()
            e.ignore()
            return
        if self._save_after_stop:
            e.ignore()
            return
        self.stop()
        super().closeEvent(e)

    def reject(self):
        if self.recording:
            self.toggle_record()
            return
        if self._save_after_stop:
            return
        self.stop()
        super().reject()
