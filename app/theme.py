"""AutoGuard AI design tokens and Qt stylesheet.

Brand: deep navy (instrument-panel feel) + one confident blue for actions; cyan only as an accent
(it is the "reading point" of the logo). Status colours are deeper and calmer than traffic-light
defaults, so a red finding reads as serious, not alarming. Same tokens as the Figma design system.
"""
from __future__ import annotations

import pyqtgraph as pg

# neutrals
NAVY = "#0B1626"          # header / brand surface
NAVY_2 = "#13233A"        # hover on navy
NAVY_LINE = "#22344F"     # separators on navy
ON_NAVY = "#E6EDF7"       # text on navy
ON_NAVY_MUTED = "#8FA3BF"
BG = "#F3F5F9"
CARD = "#FFFFFF"
SURFACE_2 = "#F7F9FC"     # table headers, subtle fills
BORDER = "#E3E8EF"
BORDER_STRONG = "#CBD3DF"
TEXT = "#0F1B2D"
MUTED = "#5B6779"
# brand / action
PRIMARY = "#1C4FD8"
PRIMARY_DARK = "#163FAF"
PRIMARY_SOFT = "#EAF0FD"
ACCENT = "#22D3EE"
# status
OK = "#0F7B55"
OK_SOFT = "#E6F4EE"
ATTENTION = "#B45309"
ATTENTION_SOFT = "#FDF3E4"
WARNING = "#C2283A"
WARNING_SOFT = "#FCEBED"
SYNTHETIC = "#6D4AE0"
SYNTHETIC_SOFT = "#F0ECFD"

ORIGIN_COLORS = {"recorded": PRIMARY, "synthetic": SYNTHETIC, "live_obd": OK}
SEVERITY_COLORS = {"attention": ATTENTION, "warning": WARNING}
LEVEL_COLORS = {"ok": (OK, OK_SOFT), "attention": (ATTENTION, ATTENTION_SOFT),
                "warning": (WARNING, WARNING_SOFT), "none": (MUTED, "#EDF0F5")}
STATUS_COLORS = {"Normal": OK, "Attention": ATTENTION, "Warning": WARNING}
PLOT_COLORS = ["#1C4FD8", "#0E9F8E", "#D97706", "#C2283A", "#6D4AE0", "#0891B2", "#65A30D", "#BE185D"]

QSS = f"""
* {{ font-family: 'Segoe UI', 'Tahoma'; }}
QWidget {{ background: {BG}; color: {TEXT}; font-size: 10.5pt; }}
QDialog {{ background: {BG}; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#card QLabel, QFrame#card QCheckBox {{ background: transparent; }}

/* ---- navy header ---- */
QFrame#header {{ background: {NAVY}; border: none; }}
QFrame#header QLabel {{ background: transparent; color: {ON_NAVY}; }}
QLabel#brand {{ font-size: 15pt; font-weight: 700; color: white; letter-spacing: 0.3px; }}
QFrame#header QLabel#brandAi {{ font-size: 9pt; font-weight: 800; color: {NAVY}; background: {ACCENT};
                  border-radius: 6px; padding: 1px 6px; }}
QPushButton#nav {{ background: transparent; border: none; border-radius: 0; padding: 18px 16px 15px 16px;
                   font-size: 11pt; font-weight: 600; color: {ON_NAVY_MUTED}; border-bottom: 3px solid transparent; }}
QPushButton#nav:hover {{ color: white; }}
QPushButton#nav:checked {{ color: white; border-bottom: 3px solid {ACCENT}; }}
QPushButton#onNavy {{ background: transparent; border: 1px solid {NAVY_LINE}; border-radius: 8px;
                      padding: 7px 14px; color: {ON_NAVY}; }}
QPushButton#onNavy:hover {{ background: {NAVY_2}; border-color: {ON_NAVY_MUTED}; color: white; }}

QLabel#h1 {{ font-size: 21pt; font-weight: 700; color: {TEXT}; }}
QLabel#h2 {{ font-size: 13pt; font-weight: 700; color: {TEXT}; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#eyebrow {{ color: {PRIMARY}; font-size: 9pt; font-weight: 700; letter-spacing: 1px; }}

QPushButton {{ background: {CARD}; border: 1px solid {BORDER_STRONG}; border-radius: 8px; padding: 8px 16px; color: {TEXT};
               font-weight: 600; }}
QPushButton:hover {{ border-color: {PRIMARY}; color: {PRIMARY}; background: {PRIMARY_SOFT}; }}
QPushButton:pressed {{ background: #DCE6FB; }}
QPushButton:disabled {{ color: #A6B0BF; background: #F1F3F7; border-color: {BORDER}; }}
QPushButton#primary {{ background: {PRIMARY}; border: 1px solid {PRIMARY}; color: white; }}
QPushButton#primary:hover {{ background: {PRIMARY_DARK}; border-color: {PRIMARY_DARK}; color: white; }}
QPushButton#primary:pressed {{ background: #11348F; }}
QPushButton#primary:disabled {{ background: #A9BCEB; border-color: #A9BCEB; color: white; }}
QPushButton#danger {{ color: {WARNING}; border-color: #EBC2C8; }}
QPushButton#danger:hover {{ background: {WARNING_SOFT}; border-color: {WARNING}; color: {WARNING}; }}
QPushButton#link {{ background: transparent; border: none; color: {PRIMARY}; padding: 4px; text-align: left;
                    font-weight: 600; }}
QPushButton#link:hover {{ text-decoration: underline; background: transparent; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {CARD}; border: 1px solid {BORDER_STRONG};
    border-radius: 8px; padding: 7px 10px; }}
QComboBox:hover, QLineEdit:focus, QSpinBox:focus {{ border-color: {PRIMARY}; }}
QComboBox::drop-down {{ border: none; width: 26px; subcontrol-origin: padding; }}
QComboBox::down-arrow {{ image: url(__ARROW__); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {CARD}; border: 1px solid {BORDER}; selection-background-color: {PRIMARY_SOFT};
    selection-color: {TEXT}; outline: none; }}
QTableWidget, QTableView, QTextEdit, QPlainTextEdit, QListWidget {{ background: {CARD}; border: 1px solid {BORDER};
    border-radius: 10px; gridline-color: {BORDER}; selection-background-color: {PRIMARY_SOFT}; selection-color: {TEXT};
    alternate-background-color: {SURFACE_2}; }}
QHeaderView::section {{ background: {SURFACE_2}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 8px 6px; font-weight: 700; font-size: 9.5pt; }}
QScrollArea {{ border: none; background: {BG}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER_STRONG}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #AEB9C9; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_STRONG}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QSlider::groove:horizontal {{ height: 6px; background: {BORDER}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {PRIMARY}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; border: 2px solid {PRIMARY}; width: 14px; margin: -6px 0; border-radius: 9px; }}
QProgressBar {{ background: {BORDER}; border: none; border-radius: 6px; height: 12px; text-align: center;
    color: {TEXT}; font-weight: 600; }}
QProgressBar::chunk {{ background: {PRIMARY}; border-radius: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QGroupBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; margin-top: 16px; padding: 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; right: 12px; padding: 0 4px; color: {MUTED}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 10px; background: {CARD}; }}
QTabBar::tab {{ background: transparent; padding: 8px 14px; color: {MUTED}; font-weight: 600; }}
QTabBar::tab:selected {{ color: {PRIMARY}; border-bottom: 2px solid {PRIMARY}; }}
QStatusBar {{ background: {CARD}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QToolTip {{ background: {NAVY}; color: white; border: none; padding: 6px 8px; }}
"""


def _arrow_path() -> str:
    from core.paths import resource_root
    return (resource_root() / "app" / "assets" / "chevron_down.svg").as_posix()


QSS = QSS.replace("__ARROW__", _arrow_path())


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(antialias=True, background=CARD, foreground=MUTED)
