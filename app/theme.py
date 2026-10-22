"""Dark automotive theme."""
from __future__ import annotations

import pyqtgraph as pg

BG = "#12161c"
PANEL = "#1b2129"
PANEL_2 = "#232b35"
BORDER = "#2f3a47"
TEXT = "#e3e8ee"
MUTED = "#8b97a6"
ACCENT = "#29b6f6"
OK = "#43a047"
ATTENTION = "#fb8c00"
WARNING = "#e53935"
SYNTHETIC = "#ab47bc"

ORIGIN_COLORS = {"recorded": ACCENT, "synthetic": SYNTHETIC, "live_obd": OK}
SEVERITY_COLORS = {"attention": ATTENTION, "warning": WARNING}
STATUS_COLORS = {"Normal": OK, "Attention": ATTENTION, "Warning": WARNING}

PLOT_COLORS = ["#29b6f6", "#66bb6a", "#ffa726", "#ef5350", "#ab47bc", "#26c6da", "#d4e157", "#ec407a"]

QSS = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI'; font-size: 10pt; }}
QFrame#panel, QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QGroupBox {{ margin-top: 14px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QListWidget#nav {{ background: {PANEL}; border: none; border-right: 1px solid {BORDER}; padding-top: 8px; }}
QListWidget#nav::item {{ padding: 10px 14px; border-radius: 4px; margin: 2px 6px; }}
QListWidget#nav::item:selected {{ background: {PANEL_2}; color: {ACCENT}; }}
QPushButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #55606d; }}
QPushButton#primary {{ background: #0277bd; border-color: #0288d1; }}
QPushButton#danger {{ background: #6d1b1b; border-color: #b71c1c; }}
QTableWidget, QTableView, QTreeWidget, QTextEdit, QPlainTextEdit {{
    background: {PANEL}; border: 1px solid {BORDER}; gridline-color: {BORDER};
    selection-background-color: #0d47a1; }}
QHeaderView::section {{ background: {PANEL_2}; color: {MUTED}; border: none; padding: 4px; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {PANEL_2}; border: 1px solid {BORDER};
    border-radius: 4px; padding: 3px 6px; }}
QSlider::groove:horizontal {{ height: 6px; background: {PANEL_2}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QLabel#h1 {{ font-size: 16pt; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QTabBar::tab {{ background: {PANEL_2}; padding: 6px 12px; border: 1px solid {BORDER}; }}
QTabBar::tab:selected {{ color: {ACCENT}; }}
"""


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(antialias=False, background=PANEL, foreground=MUTED)
