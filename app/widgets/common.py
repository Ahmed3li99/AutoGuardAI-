"""Reusable widgets for the light UI."""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from PySide6.QtGui import QColor

from app import theme
from app.i18n import pick, tr
from core.schema import CANONICAL_FIELDS


def h1(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("h1")
    return lab


def h2(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("h2")
    return lab


def muted(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("muted")
    lab.setWordWrap(True)
    return lab


def wrap(text: str, size: float | None = None, bold: bool = False, color: str | None = None) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    css = []
    if size:
        css.append(f"font-size:{size}pt")
    if bold:
        css.append("font-weight:700")
    if color:
        css.append(f"color:{color}")
    if css:
        lab.setStyleSheet(";".join(css) + ";background:transparent")
    return lab


def button(text: str, kind: str = "", slot=None) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    b.setCursor(Qt.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    return b


class Card(QFrame):
    def __init__(self, parent: QWidget | None = None, shadow: bool = True, margins: int = 18):
        super().__init__(parent)
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(margins, margins, margins, margins)
        self.lay.setSpacing(8)
        if shadow:
            eff = QGraphicsDropShadowEffect(self)
            eff.setBlurRadius(18)
            eff.setOffset(0, 2)
            eff.setColor(QColor(20, 40, 80, 22))
            self.setGraphicsEffect(eff)


class Pill(QLabel):
    def __init__(self, text: str = "", fg: str = theme.MUTED, bg: str = "#EEF1F5"):
        super().__init__()
        self.set(text, fg, bg)

    def set(self, text: str, fg: str, bg: str) -> None:
        self.setText(text)
        self.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:11px; padding:4px 12px; font-weight:600;")


# Backwards-compatible name used by older widgets.
class Badge(Pill):
    def __init__(self, text: str = "", color: str = theme.MUTED):
        super().__init__(text, "white", color)

    def set(self, text: str, color: str, bg: str | None = None) -> None:  # type: ignore[override]
        if bg is not None:
            super().set(text, color, bg)
        else:
            super().set(text, "white", color)


SIGNAL_LABELS = {
    "speed_kmh": ("السرعة", "Speed"),
    "rpm": ("لفات المحرك", "Engine RPM"),
    "coolant_temp_c": ("حرارة المحرك", "Engine temp"),
    "battery_voltage": ("الفولت", "Voltage"),
    "engine_load_pct": ("حمل المحرك", "Engine load"),
    "throttle_pct": ("البوابة", "Throttle"),
    "intake_air_temp_c": ("حرارة هوا السحب", "Intake air temp"),
    "manifold_pressure_kpa": ("ضغط السحب", "Intake pressure"),
    "fuel_level_pct": ("مستوى البنزين", "Fuel level"),
    "baro_pressure_kpa": ("الضغط الجوي", "Barometric pressure"),
    "mass_air_flow_g_per_s": ("كمية الهوا الداخلة (MAF)", "Mass air flow"),
    "ambient_air_temp_c": ("حرارة الجو", "Ambient temperature"),
}


def signal_label(sig: str) -> str:
    ar, en = SIGNAL_LABELS.get(sig, (sig, sig))
    return pick(ar, en)


class ValueCard(Card):
    """Big live value for one signal; shows N/A when the signal is unavailable."""

    def __init__(self, field: str):
        super().__init__(margins=14)
        self.field = field
        self.title = muted(signal_label(field))
        self.value = QLabel(tr("na"))
        self.value.setStyleSheet("font-size:24pt; font-weight:700; background:transparent")
        self.unit = muted(CANONICAL_FIELDS[field].unit)
        row = QHBoxLayout()
        row.addWidget(self.value)
        row.addWidget(self.unit, 0, Qt.AlignBottom)
        row.addStretch()
        self.lay.addWidget(self.title)
        self.lay.addLayout(row)
        self.setMinimumWidth(150)

    def update_value(self, v: float | None, available: bool = True, color: str | None = None) -> None:
        if not available:
            self.value.setText(tr("na"))
            self.value.setStyleSheet(f"font-size:16pt; font-weight:600; color:{theme.MUTED}; background:transparent")
            return
        txt = "—" if v is None or v != v else (f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}")
        self.value.setText(txt)
        self.value.setStyleSheet(f"font-size:24pt; font-weight:700; color:{color or theme.TEXT}; background:transparent")


# Kept for the technical-details widgets.
SignalCard = ValueCard


class ClickableCard(Card):
    clicked = Signal()

    def mousePressEvent(self, e):  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(e)


def fill_table(table: QTableWidget, df: pd.DataFrame, max_rows: int | None = None,
               float_fmt: str = "{:.3g}") -> None:
    table.clear()
    df = df if max_rows is None else df.head(max_rows)
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for i, row in enumerate(df.itertuples(index=False)):
        for j, v in enumerate(row):
            if isinstance(v, float):
                txt = "" if v != v else float_fmt.format(v)
            else:
                txt = "" if v is None else str(v)
            item = QTableWidgetItem(txt)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, j, item)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)


def readonly_table() -> QTableWidget:
    t = QTableWidget()
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.verticalHeader().setVisible(False)
    t.setShowGrid(False)
    return t


def hline() -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{theme.BORDER}")
    return f


def vbox(*widgets, spacing: int = 8) -> QVBoxLayout:
    lay = QVBoxLayout()
    lay.setSpacing(spacing)
    for w in widgets:
        if isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    return lay
