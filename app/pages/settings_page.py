"""Settings: vehicle profile, alert limits, models, adapter ports, privacy (Arabic / English)."""
from __future__ import annotations

from dataclasses import fields

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ai.rules_baseline import RuleConfig
from app.i18n import pick
from app.state import AppState
from app.widgets.common import button, muted
from core.data_sources.base import DataSourceError
from core.data_sources.obd_optional import OBDDataSource
from database.connection import default_db_path

RULE_LABELS = {
    "coolant_attention_c": ("حرارة المحرك — حد الملاحظة (°م)", "Coolant — observation limit (°C)"),
    "coolant_warning_c": ("حرارة المحرك — حد التحذير (°م)", "Coolant — warning limit (°C)"),
    "coolant_min_duration_s": ("أقل مدة لتنبيه الحرارة (ثانية)", "Coolant alert minimum duration (s)"),
    "running_rpm": ("المحرك يعتبر شغال فوق (لفة/دقيقة)", "Engine counts as running above (rpm)"),
    "charging_low_attention_v": ("فولت الشحن — حد الملاحظة (فولت)", "Charging voltage — observation limit (V)"),
    "charging_low_warning_v": ("فولت الشحن — حد التحذير (فولت)", "Charging voltage — warning limit (V)"),
    "voltage_high_warning_v": ("فولت عالي — حد التحذير (فولت)", "High voltage — warning limit (V)"),
    "voltage_min_duration_s": ("أقل مدة لتنبيه الفولت (ثانية)", "Voltage alert minimum duration (s)"),
    "rpm_high_attention": ("لفات عالية — حد الملاحظة", "High RPM — observation limit"),
    "rpm_min_duration_s": ("أقل مدة لتنبيه اللفات (ثانية)", "RPM alert minimum duration (s)"),
    "stuck_window_s": ("مدة «القراءة الواقفة» (ثانية)", "Frozen-reading window (s)"),
    "stuck_speed_change_kmh": ("تغيّر السرعة المطلوب لاكتشاف القراءة الواقفة (كم/س)",
                               "Speed change needed to detect a frozen reading (km/h)"),
}


class SettingsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(12)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # ---- vehicle profile ------------------------------------------
        vbox = QGroupBox(pick("بيانات العربية (بتكتبها إنت وبتتحفظ مع الرحلات الجديدة)",
                              "Vehicle profile (entered by you, attached to new trips)"))
        vf = QFormLayout(vbox)
        self.v_name, self.v_make, self.v_model, self.v_fuel = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.v_year = QSpinBox()
        self.v_year.setRange(0, 2100)
        self.v_year.setSpecialValueText(pick("مش معروف", "unknown"))
        self.v_notes = QLineEdit()
        for lab, w in ((pick("الاسم", "Name"), self.v_name), (pick("الماركة", "Make"), self.v_make),
                       (pick("الموديل", "Model"), self.v_model), (pick("سنة الصنع", "Year"), self.v_year),
                       (pick("نوع الوقود", "Fuel"), self.v_fuel), (pick("ملاحظات", "Notes"), self.v_notes)):
            vf.addRow(lab, w)
        vf.addRow(button(pick("احفظ", "Save profile"), "primary", self.save_profile))
        lay.addWidget(vbox)

        # ---- rules -----------------------------------------------------
        rbox = QGroupBox(pick("حدود التنبيه (قيم تقريبية — عدّلها حسب عربيتك، دي مش مواصفات المصنع)",
                              "Alert limits (heuristics — adjust to your car; not manufacturer limits)"))
        rf = QFormLayout(rbox)
        self.rule_w: dict[str, QDoubleSpinBox] = {}
        cfg = state.rule_config()
        for f in fields(RuleConfig):
            w = QDoubleSpinBox()
            w.setRange(-1000, 100000)
            w.setDecimals(1)
            w.setValue(getattr(cfg, f.name))
            self.rule_w[f.name] = w
            rf.addRow(pick(*RULE_LABELS.get(f.name, (f.name, f.name))), w)
        rb = QHBoxLayout()
        rb.addWidget(button(pick("احفظ الحدود", "Save limits"), "primary", self.save_rules))
        rb.addWidget(button(pick("رجّع الافتراضي", "Reset defaults"), "", self.reset_rules))
        rb.addStretch()
        rf.addRow(rb)
        lay.addWidget(rbox)

        # ---- models ----------------------------------------------------
        mbox = QGroupBox(pick("الموديلات", "Models"))
        mf = QFormLayout(mbox)
        self.m_enable = QCheckBox(pick("استخدم موديل MOMENT (أبطأ شوية، بيشتغل من غير نت)",
                                       "Use the MOMENT model (a bit slower, works offline)"))
        self.m_enable.setChecked(bool(state.repo.get_setting("moment_enabled", True)))
        self.m_device = QComboBox()
        self.m_device.addItem(pick("تلقائي (كارت الشاشة لو موجود)", "auto (GPU if available)"), "auto")
        self.m_device.addItem(pick("المعالج (CPU)", "CPU"), "cpu")
        self.m_device.addItem(pick("كارت الشاشة (CUDA)", "GPU (CUDA)"), "cuda")
        self.m_device.setCurrentIndex(max(0, self.m_device.findData(state.repo.get_setting("moment_device", "auto"))))
        self.m_status = muted("")
        mf.addRow(self.m_enable)
        mf.addRow(pick("التشغيل على", "Run on"), self.m_device)
        mf.addRow(button(pick("طبّق", "Apply"), "primary", self.save_models))
        mf.addRow(self.m_status)
        lay.addWidget(mbox)

        # ---- hardware --------------------------------------------------
        hbox = QGroupBox(pick("محوّل OBD-II", "OBD-II adapter"))
        hl = QVBoxLayout(hbox)
        hl.addWidget(muted(pick(
            "الاتصال بيكون من «اتصل بالعربية» في الشاشة الرئيسية. الزرار ده بس بيعرض منافذ الـ COM الموجودة "
            "لو حبيت تتأكد إن ويندوز شايف المحوّل. الاتصال قراءة بس — البرنامج عمره ما بيمسح أكواد الأعطال ولا "
            "بيغيّر حاجة في العربية. لسه ماتجربش على عربية حقيقية.",
            "Connect from “Connect to the car” on the Home screen. This button only lists COM ports so you can "
            "check that Windows sees the adapter. The connection is read-only — the app never clears trouble "
            "codes or changes anything in the car. Not yet tested on a real car.")))
        self.h_status = QLabel("")
        self.h_status.setWordWrap(True)
        hl.addWidget(button(pick("اعرض المنافذ", "List serial ports"), "", self.scan_ports))
        hl.addWidget(self.h_status)
        lay.addWidget(hbox)

        # ---- privacy -----------------------------------------------------
        pbox = QGroupBox(pick("الخصوصية", "Privacy"))
        pl = QVBoxLayout(pbox)
        pl.addWidget(muted(pick(
            "البرنامج مابيتصلش بالنت خالص: مفيش تتبّع، ولا سحابة، ولا رفع داتا. كل الرحلات محفوظة على جهازك هنا:",
            "The app never connects to the internet: no tracking, no cloud, no uploads. Trips are stored only here:")))
        pl.addWidget(QLabel(str(default_db_path())))
        pl.addWidget(muted(pick("تقدر تمسح السجل من الشاشة الرئيسية.", "Delete history from the Home screen.")))
        lay.addWidget(pbox)
        lay.addStretch()
        self._load_profile()

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
        vid = self.state.repo.save_vehicle(self.v_name.text() or pick("عربيتي", "My car"), self.v_make.text(),
                                           self.v_model.text(), self.v_year.value() or None, self.v_fuel.text(),
                                           self.v_notes.text(), vehicle_id=cur)
        self.state.repo.set_setting("active_vehicle_id", vid)
        self.state.busy_changed.emit(False, pick("اتحفظت بيانات العربية.", "Vehicle profile saved."))

    def save_rules(self) -> None:
        self.state.repo.set_setting("rule_config", {k: w.value() for k, w in self.rule_w.items()})
        self.state.busy_changed.emit(False, pick("اتحفظت الحدود — حلّل الرحلة تاني عشان تطبق.",
                                                 "Limits saved — analyse again to apply."))

    def reset_rules(self) -> None:
        d = RuleConfig()
        for k, w in self.rule_w.items():
            w.setValue(getattr(d, k))
        self.save_rules()

    def save_models(self) -> None:
        self.state.repo.set_setting("moment_enabled", self.m_enable.isChecked())
        self.state.repo.set_setting("moment_device", self.m_device.currentData())
        self.state.reload_registry()
        self.m_status.setText(pick("اتحفظ. هيتطبق في التحليل الجاي.", "Saved. Applies to the next analysis."))

    def scan_ports(self) -> None:
        try:
            ports = OBDDataSource(enabled=True).list_ports()
            self.h_status.setText(pick("المنافذ: ", "Ports: ") + (", ".join(ports) or pick("مفيش", "none")))
        except DataSourceError as e:
            self.h_status.setText("✖ " + str(e))
