"""Vehicle profile, rule thresholds, model settings, privacy and optional hardware status."""
from __future__ import annotations

from dataclasses import fields

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ai.rules_baseline import RuleConfig
from app.state import AppState
from app.widgets.common import h1, muted
from core.data_sources.base import DataSourceError
from core.data_sources.obd_optional import OBDDataSource
from database.connection import default_db_path


class SettingsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        outer = QVBoxLayout(self)
        outer.addWidget(h1("Vehicle Profile & Settings"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        lay = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # ---- vehicle profile ------------------------------------------
        vbox = QGroupBox("Vehicle profile (user-entered metadata; attached to new sessions)")
        vf = QFormLayout(vbox)
        self.v_name, self.v_make, self.v_model, self.v_fuel = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.v_year = QSpinBox()
        self.v_year.setRange(0, 2100)
        self.v_year.setSpecialValueText("unknown")
        self.v_notes = QLineEdit()
        for lab, w in (("Name", self.v_name), ("Make", self.v_make), ("Model", self.v_model),
                       ("Year", self.v_year), ("Fuel", self.v_fuel), ("Notes", self.v_notes)):
            vf.addRow(lab, w)
        b = QPushButton("Save profile")
        b.clicked.connect(self.save_profile)
        vf.addRow(b)
        lay.addWidget(vbox)

        # ---- rules -----------------------------------------------------
        rbox = QGroupBox("Rule thresholds (heuristics — adjust to your vehicle; not manufacturer limits)")
        rf = QFormLayout(rbox)
        self.rule_w: dict[str, QDoubleSpinBox] = {}
        cfg = state.rule_config()
        for f in fields(RuleConfig):
            w = QDoubleSpinBox()
            w.setRange(-1000, 100000)
            w.setDecimals(2)
            w.setValue(getattr(cfg, f.name))
            self.rule_w[f.name] = w
            rf.addRow(f.name, w)
        rb = QHBoxLayout()
        b1, b2 = QPushButton("Save thresholds"), QPushButton("Reset defaults")
        b1.clicked.connect(self.save_rules)
        b2.clicked.connect(self.reset_rules)
        rb.addWidget(b1)
        rb.addWidget(b2)
        rf.addRow(rb)
        lay.addWidget(rbox)

        # ---- models ----------------------------------------------------
        mbox = QGroupBox("Models")
        mf = QFormLayout(mbox)
        self.m_enable = QCheckBox("Use MOMENT-1-small (local weights, offline)")
        self.m_enable.setChecked(bool(state.repo.get_setting("moment_enabled", True)))
        self.m_device = QComboBox()
        self.m_device.addItems(["auto", "cpu", "cuda"])
        self.m_device.setCurrentText(state.repo.get_setting("moment_device", "auto"))
        mb = QPushButton("Apply model settings")
        mb.clicked.connect(self.save_models)
        self.m_status = muted("")
        mf.addRow(self.m_enable)
        mf.addRow("Device", self.m_device)
        mf.addRow(mb)
        mf.addRow(self.m_status)
        lay.addWidget(mbox)

        # ---- hardware --------------------------------------------------
        hbox = QGroupBox("Live OBD-II (Future / Adapter Required)")
        hl = QVBoxLayout(hbox)
        hl.addWidget(muted(
            "Optional read-only connector for an ELM327-type adapter via python-OBD. STATUS: NOT TESTED ON A "
            "REAL VEHICLE — only mocked tests exist. No adapter is required for any other feature. "
            "See docs/FUTURE_OBD_INTEGRATION.md for the safe, stationary test procedure."))
        self.h_flag = QCheckBox("Enable experimental OBD-II connector (feature flag)")
        self.h_flag.setChecked(bool(state.repo.get_setting("obd_enabled", False)))
        self.h_flag.toggled.connect(self._hw_flag)
        self.h_status = QLabel("")
        self.h_status.setWordWrap(True)
        hb = QPushButton("Scan serial ports")
        hb.clicked.connect(self.scan_ports)
        hl.addWidget(self.h_flag)
        hl.addWidget(self.h_status)
        hl.addWidget(hb)
        lay.addWidget(hbox)

        # ---- privacy -----------------------------------------------------
        pbox = QGroupBox("Privacy & local data")
        pl = QVBoxLayout(pbox)
        pl.addWidget(muted("AutoGuard AI makes no network connections at runtime: no telemetry, analytics, "
                           "cloud AI or remote databases. Vehicle data stay in this local database:"))
        pl.addWidget(QLabel(str(default_db_path())))
        pl.addWidget(muted("Delete history from 'Trips & History'."))
        lay.addWidget(pbox)
        lay.addStretch()
        self._load_profile()
        self._hw_flag(self.h_flag.isChecked())

    # ---- actions ------------------------------------------------------
    def _load_profile(self) -> None:
        p = self.state.vehicle_profile()
        if p:
            self.v_name.setText(p["name"] or "")
            self.v_make.setText(p["make"] or "")
            self.v_model.setText(p["model"] or "")
            self.v_year.setValue(p["year"] or 0)
            self.v_fuel.setText(p["fuel"] or "")
            self.v_notes.setText(p["notes"] or "")

    def save_profile(self) -> None:
        cur = self.state.repo.get_setting("active_vehicle_id")
        vid = self.state.repo.save_vehicle(self.v_name.text() or "My vehicle", self.v_make.text(),
                                           self.v_model.text(), self.v_year.value() or None, self.v_fuel.text(),
                                           self.v_notes.text(), vehicle_id=cur)
        self.state.repo.set_setting("active_vehicle_id", vid)
        self.state.busy_changed.emit(False, "Vehicle profile saved.")

    def save_rules(self) -> None:
        self.state.repo.set_setting("rule_config", {k: w.value() for k, w in self.rule_w.items()})
        self.state.busy_changed.emit(False, "Rule thresholds saved — re-run analysis to apply.")

    def reset_rules(self) -> None:
        d = RuleConfig()
        for k, w in self.rule_w.items():
            w.setValue(getattr(d, k))
        self.save_rules()

    def save_models(self) -> None:
        self.state.repo.set_setting("moment_enabled", self.m_enable.isChecked())
        self.state.repo.set_setting("moment_device", self.m_device.currentText())
        self.state.reload_registry()
        self.m_status.setText("Saved. Models reload on the next analysis.")

    def _hw_flag(self, on: bool) -> None:
        self.state.repo.set_setting("obd_enabled", on)
        ok, msg = OBDDataSource(enabled=on).availability()
        self.h_status.setText(("✔ " if ok else "✖ ") + msg + "  Connection status: not connected.")

    def scan_ports(self) -> None:
        try:
            ports = OBDDataSource(enabled=self.h_flag.isChecked()).list_ports()
            self.h_status.setText(f"Serial ports found: {ports or 'none'}. Not connected "
                                  "(live streaming UI is deferred until hardware testing).")
        except DataSourceError as e:
            self.h_status.setText("✖ " + str(e))
