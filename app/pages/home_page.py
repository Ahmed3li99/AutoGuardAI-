"""Home: choose a data source (file / sample / demo / live car) and reopen past trips."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMessageBox,
                               QScrollArea, QVBoxLayout, QWidget)

from app import icons, theme
from app.i18n import lang, pick, tr
from app.state import AppState
from ai.explain import display_name
from app.widgets.common import Card, Pill, button, fill_table, h2, readonly_table, wrap
from core.data_sources.files import RecordedFileSource
from core.data_sources.synthetic_demo import SyntheticDemoDataSource
from core.paths import SAMPLE_DIR

# icon, colour, tint for each start card
ICONS = {"file": ("folder", theme.PRIMARY, theme.PRIMARY_SOFT), "sample": ("car", "#0E7C86", "#E3F5F6"),
         "demo": ("flask", theme.SYNTHETIC, theme.SYNTHETIC_SOFT), "live": ("plug", theme.OK, theme.OK_SOFT)}
SAMPLE_LABELS = {
    "toyota_carobd_long10.csv": ("رحلة طويلة على الطريق (41 دقيقة)", "Long highway trip (41 min)"),
    "toyota_carobd_ufpe7.csv": ("رحلة داخل الجامعة، سرعة قليلة (19 دقيقة)", "Campus trip, low speed (19 min)"),
    "toyota_carobd_idle3.csv": ("العربية واقفة والمحرك شغال (11 دقيقة)", "Parked, engine idling (11 min)"),
    "dieselobd_test_excerpt_3000rows.xlsx": ("عربيات ديزل — ملف Excel بدون وقت", "Diesel cars — Excel file without time"),
}


class HomePage(QWidget):
    def __init__(self, state: AppState, open_live, open_advanced, open_cars=None):
        super().__init__()
        self.state = state
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(36, 28, 36, 28)
        lay.setSpacing(18)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        lay.addWidget(self._hero())

        # ---- current car / learning status ("learn your car") ---------------------
        self.car_card = Card()
        crow = QHBoxLayout()
        crow.addWidget(icons.tile("gauge", theme.OK, theme.OK_SOFT))
        crow.addSpacing(6)
        self.car_lbl = wrap("", 11.5, True)
        self.car_sub = wrap("", 10, False, theme.MUTED)
        col = QVBoxLayout()
        col.addWidget(self.car_lbl)
        col.addWidget(self.car_sub)
        crow.addLayout(col, 1)
        if open_cars:
            crow.addWidget(button(pick("عربياتي — اتعلّم عربيتك", "My cars — learn your car"), "primary", open_cars))
        self.car_card.lay.addLayout(crow)
        lay.addWidget(self.car_card)

        grid = QGridLayout()
        grid.setSpacing(16)
        # 1. file
        c1 = self._card("file", tr("card_file_title"), tr("card_file_text"))
        c1.lay.addWidget(button(tr("card_file_btn"), "primary", self.open_file))
        c1.lay.addWidget(button(tr("advanced_import"), "link", open_advanced))
        # 2. sample
        c2 = self._card("sample", tr("card_sample_title"), tr("card_sample_text"))
        self.samples = QComboBox()
        order = list(SAMPLE_LABELS)
        files = [p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".json")
                 and p.name != "SAMPLES.json"]
        for p in sorted(files, key=lambda x: order.index(x.name) if x.name in order else len(order)):
            ar, en = SAMPLE_LABELS.get(p.name, (p.name, p.name))
            self.samples.addItem(pick(ar, en), str(p))
        c2.lay.addWidget(self.samples)
        c2.lay.addWidget(button(tr("card_sample_btn"), "", self.open_sample))
        # 3. demo
        c3 = self._card("demo", tr("card_demo_title"), tr("card_demo_text"))
        c3.lay.addWidget(Pill(pick("داتا تجريبية — مش عطل حقيقي", "Demo data — not a real fault"),
                              theme.SYNTHETIC, theme.SYNTHETIC_SOFT), 0, Qt.AlignLeft)
        c3.lay.addWidget(button(tr("card_demo_btn"), "", self.open_demo))
        # 4. live
        c4 = self._card("live", tr("card_live_title"), tr("card_live_text"))
        c4.lay.addWidget(button(tr("card_live_btn"), "", open_live))
        for i, c in enumerate((c1, c2, c3, c4)):
            grid.addWidget(c, i // 2, i % 2)
        lay.addLayout(grid)

        hist = Card()
        top = QHBoxLayout()
        top.addWidget(h2(tr("recent")))
        top.addStretch()
        top.addWidget(button(tr("open"), "primary", self.open_selected))
        top.addWidget(button(tr("delete"), "", self.delete_selected))
        top.addWidget(button(tr("delete_all"), "danger", self.delete_all))
        hist.lay.addLayout(top)
        self.table = readonly_table()
        self.table.setMinimumHeight(220)
        self.table.doubleClicked.connect(lambda _: self.open_selected())
        hist.lay.addWidget(self.table)
        lay.addWidget(hist)
        lay.addStretch()
        state.history_changed.connect(self.refresh)
        self.refresh()

    def _card(self, icon: str, title: str, text: str) -> Card:
        c = Card()
        head = QHBoxLayout()
        name, fg, bg = ICONS[icon]
        head.addWidget(icons.tile(name, fg, bg))
        head.addSpacing(6)
        head.addWidget(h2(title), 1)
        c.lay.addLayout(head)
        c.lay.addWidget(wrap(text, color=theme.MUTED))
        c.lay.addStretch()
        return c

    def _hero(self) -> QWidget:
        """Navy brand banner with the logo: what the app is, in one line."""
        hero = QWidget()
        hero.setObjectName("hero")
        hero.setAttribute(Qt.WA_StyledBackground, True)
        hero.setStyleSheet(
            "QWidget#hero { border-radius: 16px; background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
            f"stop:0 {theme.NAVY}, stop:0.65 #132B57, stop:1 #1C4FD8); }}"
            "QWidget#hero QLabel { background: transparent; }")
        row = QHBoxLayout(hero)
        row.setContentsMargins(30, 24, 30, 24)
        row.setSpacing(22)
        mark = QLabel()
        mark.setPixmap(icons.pixmap_file(icons.logo_path(128), 84))
        row.addWidget(mark)
        col = QVBoxLayout()
        col.setSpacing(6)
        eyebrow = QLabel(pick("فحص بيانات العربية بالذكاء الاصطناعي — على جهازك", "AI vehicle-data check — on your PC"))
        eyebrow.setStyleSheet(f"color:{theme.ACCENT}; font-size:9.5pt; font-weight:700; letter-spacing:0.5px;")
        title = QLabel(tr("start_title"))
        title.setStyleSheet("color:white; font-size:22pt; font-weight:700;")
        sub = QLabel(tr("tagline"))
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{theme.ON_NAVY}; font-size:11pt;")
        for w in (eyebrow, title, sub):
            col.addWidget(w)
        row.addLayout(col, 1)
        return hero

    # ---- sources ---------------------------------------------------------
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("card_file_title"), "",
                                              "Vehicle data (*.csv *.xlsx *.xls *.json *.txt)")
        if path:
            self._go(RecordedFileSource(Path(path)))

    def open_sample(self) -> None:
        p = self.samples.currentData()
        if p:
            self._go(RecordedFileSource(Path(p)))

    def open_demo(self) -> None:
        self._go(SyntheticDemoDataSource("real_trip_faults"))

    def _go(self, source) -> None:
        self.state.load_source_async(source)
        self.window().navigate("results")  # type: ignore[attr-defined]

    # ---- history -----------------------------------------------------------
    def refresh_car(self) -> None:
        from ai.personal import has_personal_model, readiness
        repo = self.state.repo
        vid = repo.get_setting("active_vehicle_id")
        car = next((v for v in repo.list_vehicles() if v["id"] == vid), None)
        if car is None:
            self.car_lbl.setText(pick("مفيش عربية محددة", "No car selected"))
            self.car_sub.setText(pick("ضيف عربيتك عشان البرنامج يتعلّم الطبيعي بتاعها من رحلاتها — لحد كده بيستخدم "
                                      "الموديل العام (متعلّم على تويوتا إتيوس).",
                                      "Add your car so the app can learn what is normal for it from its own trips — "
                                      "until then it uses the general model (learned on a Toyota Etios)."))
            return
        name = " ".join(str(x) for x in (car["name"], car["make"] or "", car["model"] or "") if x).strip()
        self.car_lbl.setText(pick("عربيتك: ", "Your car: ") + name)
        if has_personal_model(vid):
            self.car_sub.setText(pick("البرنامج اتعلّم عربيتك — رحلاتها بتتقارن بالطبيعي بتاعها هي.",
                                      "Learned — its trips are compared with what is normal for THIS car."))
        else:
            r = readiness(repo, vid)
            self.car_sub.setText(pick(
                f"لسه بيتعلّم: {r['trips']} من {r['need_trips']} رحلات مرجعية · {r['minutes']:.0f} من "
                f"{r['need_minutes']:.0f} دقيقة. لحد كده بيستخدم الموديل العام.",
                f"Still learning: {r['trips']} of {r['need_trips']} reference trips · {r['minutes']:.0f} of "
                f"{r['need_minutes']:.0f} min. Uses the general model until then."))

    def refresh(self) -> None:
        self.refresh_car()
        df = self.state.repo.list_sessions()
        self._ids = df["id"].tolist()
        origin = {"recorded": pick("مسجّلة", "Recorded"), "synthetic": pick("تجريبية", "Demo"),
                  "live_obd": pick("مباشر", "Live")}
        show = df.assign(
            **{pick("التاريخ", "Date"): df["created_at"].str.slice(0, 16),
               pick("الرحلة", "Trip"): df["source_name"].map(lambda n: display_name(n, lang())),
               pick("النوع", "Type"): df["origin"].map(origin),
               pick("المدة", "Duration"): df["duration_s"].map(lambda s: f"{int(s) // 60} min"),
               pick("التنبيهات", "Alerts"): df["n_alerts"].fillna(0).astype(int).astype(str)}
        )[[pick("التاريخ", "Date"), pick("الرحلة", "Trip"), pick("النوع", "Type"), pick("المدة", "Duration"),
           pick("التنبيهات", "Alerts")]] if len(df) else df.iloc[0:0, 0:0]
        fill_table(self.table, show)

    def _selected(self) -> int | None:
        rows = self.table.selectionModel().selectedRows()
        return int(self._ids[rows[0].row()]) if rows else None

    def open_selected(self) -> None:
        sid = self._selected()
        if sid is not None:
            self.state.open_saved_session(sid)
            self.window().navigate("results")  # type: ignore[attr-defined]

    def delete_selected(self) -> None:
        sid = self._selected()
        if sid is not None and QMessageBox.question(self, tr("delete"), tr("confirm_delete")) == QMessageBox.Yes:
            self.state.repo.delete_session(sid)
            if self.state.session_id == sid:
                self.state.session_id = None
            self.state.history_changed.emit()

    def delete_all(self) -> None:
        if QMessageBox.question(self, tr("delete_all"), tr("confirm_delete_all")) == QMessageBox.Yes:
            self.state.repo.delete_all_history()
            self.state.session_id = None
            self.state.history_changed.emit()
