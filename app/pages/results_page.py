"""Results: verdict, trip summary and plain-language findings."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget)

from ai.explain import Finding, coverage, display_name, mmss, trip_summary, verdict
from app import icons, theme
from app.i18n import lang, pick, tr
from app.state import AppState
from app.widgets.common import Card, Pill, button, h2, hline, muted, wrap

LEVEL_ICON = {"ok": "check", "attention": "info", "warning": "alert", "none": "info"}
ROW_ICONS = {"what": "results", "normal": "gauge", "meaning": "info", "action": "settings", "evidence": "shield"}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def titled(icon_name: str, text: str) -> QHBoxLayout:
    """Section heading with a small icon tile."""
    row = QHBoxLayout()
    row.setSpacing(10)
    row.addWidget(icons.tile(icon_name, theme.ACCENT_TEXT, theme.ACCENT_SOFT, 32, 8))
    row.addWidget(h2(text), 1)
    return row


class FindingCard(Card):
    def __init__(self, f: Finding, on_show):
        super().__init__()
        t = f.text(lang())
        color = theme.SEVERITY_COLORS[f.severity]
        side = "left" if lang() == "en" else "right"
        self.setStyleSheet(f"QFrame#card {{ border-{side}: 5px solid {color}; }}")
        head = QHBoxLayout()
        head.addWidget(wrap(t["title"], 13, True), 1)
        sev_bg = theme.WARNING_SOFT if f.severity == "warning" else theme.ATTENTION_SOFT
        head.addWidget(Pill(tr("sev_" + f.severity), color, sev_bg))
        conf_col = {"high": theme.OK, "medium": theme.PRIMARY, "low": theme.MUTED}[f.confidence]
        head.addWidget(Pill(tr("conf_" + f.confidence), conf_col, "#F1F4F8"))
        head.addWidget(Pill(f"{mmss(f.t_start_s)}–{mmss(f.t_end_s)}", theme.MUTED, "#F1F4F8"))
        self.lay.addLayout(head)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        r = 0
        for key in ("what", "normal", "meaning", "action", "evidence"):
            if not t.get(key):
                continue
            cell = QWidget()
            cell.setStyleSheet("background:transparent")
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 1, 0, 0)
            cl.setSpacing(7)
            ic = QLabel()
            ic.setPixmap(icons.pixmap(ROW_ICONS[key], theme.MUTED, 15))
            cl.addWidget(ic)
            lab = QLabel(tr(key))
            lab.setStyleSheet(f"color:{theme.MUTED}; font-weight:600; background:transparent")
            cl.addWidget(lab)
            cl.addStretch()
            cell.setMinimumWidth(140)
            grid.addWidget(cell, r, 0, Qt.AlignTop)
            grid.addWidget(wrap(t[key], 10.5, key == "action"), r, 1)
            r += 1
        grid.setColumnStretch(1, 1)
        self.lay.addLayout(grid)
        b = button(tr("show_chart"), "link", lambda: on_show(f))
        b.setIcon(icons.icon("trip", theme.PRIMARY, size=16))
        self.lay.addWidget(b, 0, Qt.AlignLeft if lang() == "en" else Qt.AlignRight)


class ResultsPage(QWidget):
    def __init__(self, state: AppState, open_tech, open_lab=None, open_advanced=None):
        super().__init__()
        self.state = state
        self.open_tech = open_tech
        self.open_lab = open_lab
        self.open_advanced = open_advanced
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll)
        self._show_low = False
        state.analysis_changed.connect(self.rebuild)
        state.session_changed.connect(self.rebuild)
        state.busy_changed.connect(self._busy)
        self._busy_msg = ""
        self.rebuild()

    def _busy(self, on: bool, msg: str) -> None:
        self._busy_msg = msg if on else ""
        self.rebuild()

    def rebuild(self) -> None:
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(36, 24, 36, 28)
        lay.setSpacing(16)
        s, res = self.state.session, self.state.result

        if self._busy_msg:
            c = Card()
            c.lay.addWidget(wrap(tr(self._busy_msg) if self._busy_msg in ("loading", "analysing")
                                 else self._busy_msg, 15, True))
            c.lay.addWidget(muted(pick("أول تحليل ممكن ياخد شوية عشان الموديلات بتتحمّل.",
                                       "The first analysis can take a moment while models load.")))
            lay.addWidget(c)
        if s is None:
            if not self._busy_msg:
                c = Card()
                c.lay.addWidget(wrap(tr("no_trip"), 15, True))
                c.lay.addWidget(button(tr("nav_home"), "primary", lambda: self.window().navigate("home")),
                                0, Qt.AlignLeft)
                lay.addWidget(c)
            lay.addStretch()
            self.scroll.setWidget(body)
            return

        # ---- verdict banner ---------------------------------------------
        v = verdict(self.state.findings, res, s)
        fg, bg = theme.LEVEL_COLORS[v["level"]]
        banner = Card()
        banner.setStyleSheet(f"QFrame#card {{ background:{bg}; border:1px solid {rgba(fg, 0.35)}; }}")
        row = QHBoxLayout()
        row.setSpacing(18)
        row.addWidget(icons.tile(LEVEL_ICON[v["level"]], fg, "#FFFFFF", 64, 16), 0, Qt.AlignTop)
        col = QVBoxLayout()
        col.addWidget(wrap(pick(v["ar"], v["en"]), 19, True, fg))
        col.addWidget(wrap(pick(v["ar_sub"], v["en_sub"]), 11, False, theme.TEXT))
        origin_txt = tr("origin_" + s.meta.origin.value)
        col.addWidget(wrap(f"{origin_txt} · {display_name(s.meta.source_name, lang())}", 9.5, False, theme.MUTED))
        row.addLayout(col, 1)
        btns = QVBoxLayout()
        if res is None:
            btns.addWidget(button(tr("analyse"), "primary", self.state.run_analysis_async))
        else:
            btns.addWidget(button(tr("reanalyse"), "", self.state.run_analysis_async))
        if self.open_lab is not None:
            lab_b = button(pick("معمل السيناريوهات", "Scenario lab"), "", self.open_lab)
            lab_b.setIcon(icons.icon("flask", theme.TEXT, size=16))
            btns.addWidget(lab_b)
        btns.addWidget(button(tr("tech_details"), "link", self.open_tech))
        if self.open_advanced is not None:
            btns.addWidget(button(pick("التحليل المتقدم", "Advanced analysis"), "", self.open_advanced))
        row.addLayout(btns)
        banner.lay.addLayout(row)
        lay.addWidget(banner)

        # ---- how reliable is this result ---------------------------------
        cov = Card()
        cov.lay.addLayout(titled("shield", pick("النتيجة دي موثوقة لأي درجة؟", "How reliable is this result?")))
        cg = QGridLayout()
        cg.setHorizontalSpacing(14)
        for i, c in enumerate(coverage(res, s, self.state.quality)):
            mark = {True: ("check", theme.OK), False: ("alert", theme.ATTENTION), None: ("info", theme.MUTED)}[c["ok"]]
            m = QLabel()
            m.setPixmap(icons.pixmap(mark[0], mark[1], 18, 2.0))
            m.setStyleSheet("background:transparent")
            cg.addWidget(m, i, 0, Qt.AlignTop)
            lab = QLabel(pick(c["ar"], c["en"]))
            lab.setStyleSheet(f"color:{theme.MUTED}; font-weight:600; background:transparent")
            cg.addWidget(lab, i, 1, Qt.AlignTop)
            val = wrap(pick(c["value_ar"], c["value_en"]), 10.5)
            # number-only text (e.g. "100%") would otherwise align by its own LTR direction
            val.setAlignment((Qt.AlignRight if lang() == "ar" else Qt.AlignLeft) | Qt.AlignVCenter)
            cg.addWidget(val, i, 2)
        cg.setColumnStretch(2, 1)
        cov.lay.addLayout(cg)
        lay.addWidget(cov)

        # ---- trip summary ------------------------------------------------
        summ = Card()
        summ.lay.addLayout(titled("gauge", tr("trip_summary")))
        g = QGridLayout()
        for i, (ar, en, val) in enumerate(trip_summary(s)):
            box = QVBoxLayout()
            box.addWidget(muted(pick(ar, en)))
            box.addWidget(wrap(val, 15, True))
            g.addLayout(box, 0, i)
        summ.lay.addLayout(g)
        lay.addWidget(summ)

        # ---- findings -------------------------------------------------------
        if res is not None:
            main = [f for f in self.state.findings if f.confidence != "low" and f.category == "car"]
            data = [f for f in self.state.findings if f.confidence != "low" and f.category == "data"]
            low = [f for f in self.state.findings if f.confidence == "low"]
            lay.addLayout(titled("results", tr("findings")))
            if not main:
                c = Card()
                c.lay.addWidget(wrap((tr("no_findings") if not data else pick(
                    "مفيش ملاحظات على العربية نفسها.", "No findings about the car itself.")), 12))
                lay.addWidget(c)
            for f in main:
                lay.addWidget(FindingCard(f, self._show))
            if data:
                lay.addLayout(titled("plug", pick("موثوقية القراءات — العربية تعبانة، ولا الحساس بيكدب؟",
                                                  "Reading reliability — is it the car, or the sensor?")))
                lay.addWidget(muted(pick(
                    "هنا الأوقات اللي الحساسات قالت فيها حاجات متناقضة أو مستحيلة فيزيائياً. البرنامج بيراجع الداتا "
                    "نفسها قبل ما يحكم على العربية.",
                    "Times when sensors disagreed or reported physically impossible values. The app checks the data "
                    "itself before judging the car.")))
                for f in data:
                    lay.addWidget(FindingCard(f, self._show))
            if low:
                tog = button(("▾ " if self._show_low else "▸ ") + tr("less_important", n=len(low)), "link",
                             self._toggle_low)
                lay.addWidget(tog, 0, Qt.AlignLeft if lang() == "en" else Qt.AlignRight)
                if self._show_low:
                    for f in low:
                        lay.addWidget(FindingCard(f, self._show))

        # ---- how it works + disclaimer --------------------------------------
        how = Card()
        how.lay.addLayout(titled("cpu", tr("how_it_works")))
        for k in ("how_rules", "how_consistency", "how_eva", "how_if", "how_moment"):
            how.lay.addWidget(wrap("• " + tr(k), 10.5))
        how.lay.addWidget(hline())
        how.lay.addWidget(wrap(tr("disclaimer"), 9.5, False, theme.MUTED))
        lay.addWidget(how)
        lay.addStretch()
        self.scroll.setWidget(body)

    def _toggle_low(self) -> None:
        self._show_low = not self._show_low
        self.rebuild()

    def _show(self, f: Finding) -> None:
        self.state.focus_time.emit(f.t_start_s, f.t_end_s)
        self.window().navigate("trip")  # type: ignore[attr-defined]
