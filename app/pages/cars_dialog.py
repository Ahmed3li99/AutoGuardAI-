"""'My cars': vehicle profiles, reference trips and the personal model ('learn your car')."""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLineEdit, QMessageBox, QProgressBar, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ai.explain import display_name
from ai.personal import MIN_MINUTES, MIN_TRIPS, forget_personal_model, has_personal_model, readiness, \
    train_personal_model
from app.i18n import lang, pick
from app.state import AppState, Worker
from app.widgets.common import Card, button, h2, muted, wrap


POWERTRAINS = [("", "مش معروف", "Unknown"), ("ice", "بنزين / ديزل", "Petrol / diesel"),
               ("hybrid", "هايبرد (أو plug-in)", "Hybrid (or plug-in)"), ("ev", "كهربا", "Electric")]


def powertrain_combo() -> QComboBox:
    c = QComboBox()
    for key, ar, en in POWERTRAINS:
        c.addItem(pick(ar, en), key)
    return c


class NewCarDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(pick("عربية جديدة", "New car"))
        f = QFormLayout(self)
        self.name, self.make, self.model = QLineEdit(), QLineEdit(), QLineEdit()
        self.year = QSpinBox()
        self.year.setRange(0, 2100)
        self.year.setSpecialValueText(pick("مش معروف", "unknown"))
        self.powertrain = powertrain_combo()
        for lab, w in ((pick("اسم تختاره", "Name"), self.name), (pick("الماركة", "Make"), self.make),
                       (pick("الموديل", "Model"), self.model), (pick("سنة الصنع", "Year"), self.year),
                       (pick("نوع المحرك", "Engine type"), self.powertrain)):
            f.addRow(lab, w)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)


class CarsDialog(QDialog):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.repo = state.repo
        self.setWindowTitle(pick("عربياتي — اتعلّم عربيتك", "My cars — learn your car"))
        self.resize(1100, 780)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(12)

        top = Card()
        top.lay.addWidget(h2(pick("عربياتي", "My cars")))
        top.lay.addWidget(muted(pick(
            "البرنامج بيتعلّم الطبيعي لكل عربية من رحلات إنت تأكدها إنها سليمة («رحلات مرجعية»). بعد كده بيحكم على "
            "رحلاتها الجديدة بمقارنتها بنفسها مش بعربية تانية.",
            "The app learns what is normal for each car from trips you confirm as normal ('reference trips'), then "
            "judges its new trips against itself instead of another car.")))
        row = QHBoxLayout()
        self.cars = QComboBox()
        self.cars.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.cars, 1)
        row.addWidget(button("+  " + pick("عربية جديدة", "New car"), "", self.new_car))
        self.b_active = button(pick("خليها عربيتي الحالية", "Make it my current car"), "", self.make_active)
        row.addWidget(self.b_active)
        row.addWidget(button(pick("امسح العربية", "Delete car"), "danger", self.delete_car))
        top.lay.addLayout(row)
        prow = QHBoxLayout()
        prow.addWidget(muted(pick("نوع المحرك:", "Engine type:")))
        self.powertrain = powertrain_combo()
        self.powertrain.currentIndexChanged.connect(self.set_powertrain)
        prow.addWidget(self.powertrain)
        prow.addWidget(muted(pick("(الهايبرد بيطفي المحرك وهو ماشي، فالبرنامج مش هيعتبر «لفات صفر وهي ماشية» تناقض)",
                                  "(hybrids stop the engine while driving, so '0 RPM while moving' is not flagged)")), 1)
        top.lay.addLayout(prow)
        self.active_lbl = muted("")
        top.lay.addWidget(self.active_lbl)
        lay.addWidget(top)

        learn = Card()
        learn.lay.addWidget(h2(pick("التعلّم", "Learning")))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        learn.lay.addWidget(self.progress)
        self.learn_lbl = wrap("", 10.5)
        learn.lay.addWidget(self.learn_lbl)
        lr = QHBoxLayout()
        self.b_learn = button(pick("اتعلّم عربيتي دلوقتي", "Learn my car now"), "primary", self.learn)
        self.b_forget = button(pick("ارجع للموديل العام", "Back to the general model"), "", self.forget)
        lr.addWidget(self.b_learn)
        lr.addWidget(self.b_forget)
        lr.addStretch()
        learn.lay.addLayout(lr)
        self.model_lbl = wrap("", 10.5)
        learn.lay.addWidget(self.model_lbl)
        lay.addWidget(learn)

        trips = Card()
        trips.lay.addWidget(h2(pick("الرحلات", "Trips")))
        trips.lay.addWidget(muted(pick(
            "علّم على الرحلات اللي تبع العربية دي، وبعدين على اللي إنت متأكد إنها كانت عادية وسليمة. "
            "رحلة فيها تحذير مؤكد مينفعش تبقى مرجعية — عشان البرنامج مايتعلمش مشكلة على إنها طبيعي.",
            "Tick the trips that belong to this car, then the ones you are sure were normal. A trip with a "
            "confirmed warning cannot be a reference, so the app never learns a problem as 'normal'.")))
        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        trips.lay.addWidget(self.table, 1)
        lay.addWidget(trips, 1)
        bb = QHBoxLayout()
        bb.addStretch()
        bb.addWidget(button(pick("قفل", "Close"), "", self.accept))
        lay.addLayout(bb)
        self._fill_cars()

    # ---- cars --------------------------------------------------------------
    def _fill_cars(self, select: int | None = None) -> None:
        self.cars.blockSignals(True)
        self.cars.clear()
        active = self.repo.get_setting("active_vehicle_id")
        for v in self.repo.list_vehicles():
            label = " ".join(str(x) for x in (v["name"], v["make"] or "", v["model"] or "", v["year"] or "") if x).strip()
            self.cars.addItem(("★ " if v["id"] == active else "") + label, v["id"])
        idx = self.cars.findData(select if select is not None else active)
        self.cars.setCurrentIndex(max(idx, 0))
        self.cars.blockSignals(False)
        self.refresh()

    def vid(self) -> int | None:
        return self.cars.currentData()

    def new_car(self) -> None:
        d = NewCarDialog(self)
        if d.exec() == QDialog.Accepted:
            vid = self.repo.save_vehicle(d.name.text() or pick("عربيتي", "My car"), d.make.text(), d.model.text(),
                                         d.year.value() or None, fuel=d.powertrain.currentData() or "")
            if self.repo.get_setting("active_vehicle_id") is None:
                self.repo.set_setting("active_vehicle_id", vid)
            self._fill_cars(vid)

    def set_powertrain(self) -> None:
        if self.vid() is not None:
            self.repo.set_vehicle_fuel(self.vid(), self.powertrain.currentData() or "")
            self.state.reload_registry()

    def make_active(self) -> None:
        if self.vid() is not None:
            self.repo.set_setting("active_vehicle_id", self.vid())
            self._fill_cars(self.vid())

    def delete_car(self) -> None:
        vid = self.vid()
        if vid is None:
            return
        if QMessageBox.question(self, pick("امسح العربية", "Delete car"),
                                pick("تمسح العربية والموديل اللي اتعلمه؟ الرحلات نفسها هتفضل.",
                                     "Delete this car and its learned model? The trips themselves are kept.")) \
                != QMessageBox.Yes:
            return
        forget_personal_model(self.repo, vid)
        self.repo.delete_vehicle(vid)
        if self.repo.get_setting("active_vehicle_id") == vid:
            self.repo.set_setting("active_vehicle_id", None)
        self.state.reload_registry()
        self._fill_cars()

    # ---- state ---------------------------------------------------------------
    def refresh(self) -> None:
        vid = self.vid()
        has = vid is not None
        for b in (self.b_active, self.b_learn, self.b_forget, self.powertrain):
            b.setEnabled(has)
        self.powertrain.blockSignals(True)
        self.powertrain.setCurrentIndex(max(0, self.powertrain.findData(self.repo.vehicle_fuel(vid))))
        self.powertrain.blockSignals(False)
        self.active_lbl.setText(pick("مفيش عربيات لسه — اعمل عربية جديدة.", "No cars yet — add one.") if not has else
                                pick("الرحلات الجديدة اللي هتفتحها بتتسجل تلقائي على عربيتك الحالية (★).",
                                     "New trips you open are assigned automatically to your current car (★)."))
        if not has:
            self.table.setRowCount(0)
            self.progress.setValue(0)
            self.learn_lbl.setText("")
            self.model_lbl.setText("")
            return
        r = readiness(self.repo, vid)
        self.progress.setValue(int(100 * r["progress"]))
        self.progress.setFormat(pick(f"{r['trips']} من {MIN_TRIPS} رحلات · {r['minutes']:.0f} من {MIN_MINUTES:.0f} دقيقة",
                                     f"{r['trips']} of {MIN_TRIPS} trips · {r['minutes']:.0f} of {MIN_MINUTES:.0f} min"))
        if r["ready"]:
            msg = pick("جاهز تتعلّم ✓", "Ready to learn ✓")
            if r["trips"] < r["recommended_trips"]:
                msg += pick(f" — والنتيجة بتبقى أدق بكتير مع {r['recommended_trips']} رحلات.",
                            f" — results get much more reliable with {r['recommended_trips']} trips.")
        elif r["driving_ok"] is False:
            msg = pick(f"رحلات التعلّم فيها {r['train_moving_minutes']:.0f} دقيقة سواقة بس (فوق 20 كم/س)، ومحتاج "
                       f"{r['need_moving_minutes']:.0f} على الأقل. ضيف رحلة فيها سواقة حقيقية، مش وقوف بس — "
                       "لو اتعلّم من العربية وهي واقفة بس هيتلخبط لما تمشي.",
                       f"The learning trips contain only {r['train_moving_minutes']:.0f} min of driving above 20 km/h "
                       f"(need {r['need_moving_minutes']:.0f}). Add a trip with real driving, not only idling.")
        else:
            msg = pick(f"محتاج على الأقل {MIN_TRIPS} رحلات مرجعية و{MIN_MINUTES:.0f} دقيقة (والأفضل "
                       f"{r['recommended_trips']} رحلات فيها سواقة حقيقية).",
                       f"Needs at least {MIN_TRIPS} reference trips and {MIN_MINUTES:.0f} min (ideally "
                       f"{r['recommended_trips']} trips with real driving).")
        self.learn_lbl.setText(msg)
        self.b_learn.setEnabled(r["ready"])
        info = self.repo.vehicle_model(vid)
        self.b_forget.setEnabled(has_personal_model(vid))
        if info and has_personal_model(vid):
            from ai.explain import NAMES
            typ = pick("، ", ", ").join(
                f"{pick(NAMES[t][0], NAMES[t][1]) if t in NAMES else t} {lo:g}–{hi:g}"
                for t, rng in info["typical"].items() if rng and t != "speed_kmh" for lo, hi in [rng])
            self.model_lbl.setText(pick(
                f"العربية دي ليها موديل خاص (اتعلّم {info['trained_at'][:16].replace('T', ' ')}) من "
                f"{info['train_trips']} رحلات للتعلّم و{info['calibration_trips']} للضبط ({info['train_minutes']:.0f}+"
                f"{info['calibration_minutes']:.0f} دقيقة). على رحلات الضبط: الموديل العام كان هيطلّع "
                f"{info['base_alerts_per_hour_on_calibration']} تنبيه/ساعة، والخاص {info['personal_alerts_per_hour_on_calibration']}. "
                f"المعتاد لعربيتك: {typ}",
                f"This car has its own model (learned {info['trained_at'][:16].replace('T', ' ')}) from "
                f"{info['train_trips']} learning + {info['calibration_trips']} calibration trips "
                f"({info['train_minutes']:.0f}+{info['calibration_minutes']:.0f} min). On the calibration trips the general "
                f"model would raise {info['base_alerts_per_hour_on_calibration']} alerts/h, the personal one "
                f"{info['personal_alerts_per_hour_on_calibration']}. Usual for your car: {typ}"))
            diff = describe_car_map(info.get("car_map") or {})
            if diff:
                self.model_lbl.setText(self.model_lbl.text() + "\n" + diff)
        else:
            self.model_lbl.setText(pick("العربية دي لسه بتستخدم الموديل العام (متعلّم على تويوتا إتيوس).",
                                        "This car still uses the general model (learned on a Toyota Etios)."))
        self._fill_trips(vid)

    def _fill_trips(self, vid: int) -> None:
        df = self.repo.list_sessions()
        heads = [pick("الرحلة", "Trip"), pick("التاريخ", "Date"), pick("المدة", "Duration"), pick("التحليل", "Analysis"),
                 pick("تبع العربية دي", "Belongs to this car"), pick("رحلة مرجعية سليمة", "Normal reference trip")]
        self.table.clear()
        self.table.setColumnCount(len(heads))
        self.table.setHorizontalHeaderLabels(heads)
        self.table.setRowCount(len(df))
        for i, r in enumerate(df.itertuples()):
            sid = int(r.id)
            assigned = not pd.isna(r.vehicle_profile_id)
            mine = assigned and int(r.vehicle_profile_id) == vid
            other = assigned and not mine
            for j, txt in enumerate((display_name(r.source_name, lang()), str(r.created_at)[:16],
                                     f"{int(r.duration_s) // 60} min")):
                it = QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, j, it)
            if pd.isna(r.last_run_id):
                status = pick("لسه ماتحللتش", "not analysed")
            elif r.n_rule_warnings:
                status = pick(f"{int(r.n_rule_warnings)} تحذير", f"{int(r.n_rule_warnings)} warning(s)")
            else:
                status = pick("✓ مفيش تحذيرات", "✓ no warnings")
            it = QTableWidgetItem(status)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 3, it)
            cb_car = QCheckBox(pick("(عربية تانية)", "(other car)") if other else "")
            cb_car.setChecked(bool(mine))
            cb_car.toggled.connect(lambda on, s=sid: self._assign(s, on))
            self.table.setCellWidget(i, 4, _centered(cb_car))
            cb_ref = QCheckBox()
            cb_ref.setChecked(bool(mine and r.is_reference))
            allowed = bool(mine) and not pd.isna(r.last_run_id) and not r.n_rule_warnings
            cb_ref.setEnabled(allowed or bool(mine and r.is_reference))
            if mine and r.n_rule_warnings:
                cb_ref.setToolTip(pick("فيها تحذير مؤكد — مينفعش تبقى مرجعية", "Has a confirmed warning — cannot be a reference"))
            cb_ref.toggled.connect(lambda on, s=sid: self._reference(s, on))
            self.table.setCellWidget(i, 5, _centered(cb_ref))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _assign(self, sid: int, on: bool) -> None:
        self.repo.set_session_vehicle(sid, self.vid() if on else None)
        self.refresh()

    def _reference(self, sid: int, on: bool) -> None:
        try:
            self.repo.set_reference(sid, on)
        except ValueError as e:
            msg = {"confirmed warning": pick("الرحلة دي فيها تحذير مؤكد، فمينفعش تبقى مرجعية — وإلا البرنامج هيتعلم "
                                             "المشكلة على إنها طبيعي.", str(e)),
                   "Analyse": pick("حلّل الرحلة الأول (افتحها من الرئيسية).", str(e)),
                   "Assign": pick("علّم إن الرحلة تبع العربية دي الأول.", str(e))}
            QMessageBox.warning(self, pick("مينفعش", "Not allowed"),
                                next((v for k, v in msg.items() if k in str(e)), str(e)))
        self.refresh()

    # ---- learning ------------------------------------------------------------
    def learn(self) -> None:
        vid = self.vid()
        self.b_learn.setEnabled(False)
        self.learn_lbl.setText(pick("بيتعلّم عربيتك… (أقل من دقيقة)", "Learning your car… (under a minute)"))
        w = Worker(train_personal_model, self.repo, vid)
        w.signals.finished.connect(self._learned)
        w.signals.failed.connect(self._learn_failed)
        self.state.pool.start(w)

    def _learned(self, info: dict) -> None:
        self.state.reload_registry()
        self.refresh()
        QMessageBox.information(self, pick("تم", "Done"), pick(
            "اتعلّمت عربيتك ✓ — الرحلات الجديدة بتاعتها هتتحكم عليها بالموديل الخاص. لو عايز تطبقه على رحلة قديمة "
            "افتحها ودوس «حلّل تاني».",
            "Your car has been learned ✓ — its new trips will be judged with the personal model. To apply it to an "
            "older trip, open it and press “Analyse again”."))

    def _learn_failed(self, msg: str) -> None:
        self.refresh()
        QMessageBox.warning(self, pick("مقدرتش أتعلّم", "Could not learn"), msg)

    def forget(self) -> None:
        forget_personal_model(self.repo, self.vid())
        self.state.reload_registry()
        self.refresh()


# reference points of the base car (Toyota Etios, warm engine) used to put the learned map into words
_REF = {"coolant_temp_c": 88.0, "battery_voltage": 13.7, "intake_air_temp_c": 45.0, "rpm": 2000.0}


def describe_car_map(car_map: dict) -> str:
    """'Compared with the general model: your car runs 5.6 C hotter, ~12 % higher RPM, ...'"""
    parts_ar, parts_en = [], []
    for sig, ref in _REF.items():
        if sig not in car_map:
            continue
        a, b = car_map[sig]
        yours = a + b * ref
        if sig == "rpm":
            pct = 100 * (yours / ref - 1)
            if abs(pct) >= 3:
                parts_ar.append(f"لفاتها {'أعلى' if pct > 0 else 'أقل'} حوالي {abs(pct):.0f}% في نفس السرعة")
                parts_en.append(f"~{abs(pct):.0f}% {'higher' if pct > 0 else 'lower'} RPM at the same speed")
            continue
        d = yours - ref
        unit_ar, unit_en, thr = {"coolant_temp_c": ("°م", "°C", 1.0), "intake_air_temp_c": ("°م", "°C", 2.0),
                                 "battery_voltage": ("فولت", "V", 0.1)}[sig]
        if abs(d) >= thr:
            name_ar, name_en = {"coolant_temp_c": ("حرارة محركها", "engine runs"),
                                "intake_air_temp_c": ("حرارة هوا السحب", "intake air is"),
                                "battery_voltage": ("فولت الشحن", "charging voltage is")}[sig]
            parts_ar.append(f"{name_ar} {'أعلى' if d > 0 else 'أقل'} بحوالي {abs(d):.1f} {unit_ar}")
            word = ("hotter" if d > 0 else "cooler") if sig != "battery_voltage" else ("higher" if d > 0 else "lower")
            parts_en.append(f"{name_en} {abs(d):.1f} {unit_en} {word}")
    if not parts_ar:
        return pick("الطبيعي بتاع عربيتك قريب جداً من الموديل العام.", "Your car's normal is very close to the general model.")
    return pick("مقارنةً بالموديل العام: " + "، ".join(parts_ar) + ".",
                "Compared with the general model: " + ", ".join(parts_en) + ".")


def _centered(w: QWidget) -> QWidget:
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setAlignment(Qt.AlignCenter)
    lay.addWidget(w)
    return box

