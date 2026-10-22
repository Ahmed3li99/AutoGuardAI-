"""Small reusable widgets."""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QHeaderView, QLabel, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from app import theme
from core.schema import CANONICAL_FIELDS


class Badge(QLabel):
    def __init__(self, text: str = "", color: str = theme.MUTED):
        super().__init__()
        self.set(text, color)

    def set(self, text: str, color: str) -> None:
        self.setText(text)
        self.setStyleSheet(f"background:{color}; color:white; border-radius:9px; padding:3px 10px; "
                           "font-weight:600;")


class SignalCard(QFrame):
    """Shows one signal's latest value, or N/A if the signal is unavailable."""

    def __init__(self, field: str):
        super().__init__()
        self.setObjectName("panel")
        self.field = field
        spec = CANONICAL_FIELDS[field]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        self.title = QLabel(spec.description)
        self.title.setObjectName("muted")
        self.value = QLabel("N/A")
        self.value.setStyleSheet("font-size: 20pt; font-weight: 600;")
        self.unit = QLabel(spec.unit)
        self.unit.setObjectName("muted")
        for w in (self.title, self.value, self.unit):
            lay.addWidget(w)
        self.setMinimumWidth(150)

    def update_value(self, v: float | None, available: bool = True) -> None:
        if not available:
            self.value.setText("N/A")
            self.value.setStyleSheet(f"font-size: 20pt; font-weight: 600; color:{theme.MUTED};")
            return
        self.value.setStyleSheet("font-size: 20pt; font-weight: 600;")
        self.value.setText("—" if v is None or v != v else f"{v:,.1f}" if abs(v) < 1000 else f"{v:,.0f}")


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
    t.setAlternatingRowColors(False)
    t.verticalHeader().setVisible(False)
    return t


def h1(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("h1")
    return lab


def muted(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("muted")
    lab.setWordWrap(True)
    return lab
