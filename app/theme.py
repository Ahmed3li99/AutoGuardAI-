"""AutoGuard AI design tokens and Qt stylesheet — "Black Box" identity.

The app is the car's own black box: it records a trip and understands it. Graphite surfaces like a
recorder's casing, recorder orange as the single brand accent (active tab, REC tag, logo), warm
paper for content. Actions are graphite, not a second bright colour, so orange stays special.
Status colours are deep and calm; orange text uses a darker shade so it passes contrast (4.5:1).
Type: IBM Plex Sans Arabic (Arabic + Latin), Archivo for the wordmark, IBM Plex Mono for tags.
"""
from __future__ import annotations

import pyqtgraph as pg

# graphite (header, brand surfaces)
INK = "#121417"
INK_2 = "#1E2126"         # hover on ink
INK_LINE = "#30343B"      # separators / outlines on ink
ON_INK = "#ECEAE6"
ON_INK_MUTED = "#9A9DA3"
# paper (content)
BG = "#F6F4EF"
CARD = "#FFFFFF"
SURFACE_2 = "#FAF9F6"     # table headers, subtle fills
BORDER = "#E5E1D8"
BORDER_STRONG = "#CFCAC0"
TEXT = "#121417"
MUTED = "#5E6168"
# brand + actions
ACCENT = "#FF6A13"        # recorder orange: fills, bars, logo (not for small text)
ACCENT_TEXT = "#B8470B"   # orange for text/links on paper (contrast >= 4.5:1)
ACCENT_SOFT = "#FFF0E6"
PRIMARY = INK             # primary buttons are graphite
PRIMARY_DARK = "#2C3037"
PRIMARY_SOFT = "#EFEDE7"
# status
OK = "#1F7A4D"
OK_SOFT = "#E8F3EC"
ATTENTION = "#A16207"
ATTENTION_SOFT = "#FBF3E2"
WARNING = "#B42318"
WARNING_SOFT = "#FCEBE9"
SYNTHETIC = "#6D4AE0"
SYNTHETIC_SOFT = "#F0ECFD"
# older names still used by some widgets
NAVY, NAVY_2, NAVY_LINE, ON_NAVY, ON_NAVY_MUTED = INK, INK_2, INK_LINE, ON_INK, ON_INK_MUTED

FONT_UI = "IBM Plex Sans Arabic"
FONT_BRAND = "Archivo"
FONT_MONO = "IBM Plex Mono"

ORIGIN_COLORS = {"recorded": PRIMARY, "synthetic": SYNTHETIC, "live_obd": OK}
SEVERITY_COLORS = {"attention": ATTENTION, "warning": WARNING}
LEVEL_COLORS = {"ok": (OK, OK_SOFT), "attention": (ATTENTION, ATTENTION_SOFT),
                "warning": (WARNING, WARNING_SOFT), "none": (MUTED, "#EFEDE7")}
STATUS_COLORS = {"Normal": OK, "Attention": ATTENTION, "Warning": WARNING}
PLOT_COLORS = ["#121417", "#FF6A13", "#0E7C86", "#B42318", "#6D4AE0", "#A16207", "#1F7A4D", "#5E6168"]

QSS = f"""
* {{ font-family: '{FONT_UI}', 'Segoe UI', 'Tahoma'; }}
QWidget {{ background: {BG}; color: {TEXT}; font-size: 10.5pt; }}
QDialog {{ background: {BG}; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#card QLabel, QFrame#card QCheckBox {{ background: transparent; }}

/* ---- graphite header ---- */
QFrame#header {{ background: {NAVY}; border: none; }}
QFrame#header QLabel {{ background: transparent; color: {ON_NAVY}; }}
QLabel#brand {{ font-family: '{FONT_BRAND}'; font-size: 15pt; font-weight: 800; color: white; letter-spacing: 1.5px; }}
QFrame#header QLabel#brandAi {{ font-family: '{FONT_MONO}'; font-size: 8.5pt; font-weight: 500; color: {INK};
                  background: {ACCENT}; border-radius: 4px; padding: 2px 7px; }}
QPushButton#nav {{ background: transparent; border: none; border-radius: 0; padding: 18px 16px 15px 16px;
                   font-size: 11pt; font-weight: 600; color: {ON_NAVY_MUTED}; border-bottom: 3px solid transparent; }}
QPushButton#nav:hover {{ color: white; }}
QPushButton#nav:checked {{ color: white; border-bottom: 3px solid {ACCENT}; }}
QPushButton#onNavy {{ background: transparent; border: 1px solid {NAVY_LINE}; border-radius: 6px;
                      padding: 7px 14px; color: {ON_NAVY}; }}
QPushButton#onNavy:hover {{ background: {NAVY_2}; border-color: {ON_NAVY_MUTED}; color: white; }}

QLabel#h1 {{ font-size: 21pt; font-weight: 700; color: {TEXT}; }}
QLabel#h2 {{ font-size: 13pt; font-weight: 700; color: {TEXT}; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#eyebrow {{ color: {ACCENT_TEXT}; font-size: 9pt; font-weight: 700; letter-spacing: 1px; }}

QPushButton {{ background: {CARD}; border: 1px solid {BORDER_STRONG}; border-radius: 6px; padding: 8px 16px; color: {TEXT};
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
QPushButton#link {{ background: transparent; border: none; color: {ACCENT_TEXT}; padding: 4px; text-align: left;
                    font-weight: 600; }}
QPushButton#link:hover {{ text-decoration: underline; background: transparent; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {CARD}; border: 1px solid {BORDER_STRONG};
    border-radius: 6px; padding: 7px 10px; }}
QComboBox:hover, QLineEdit:focus, QSpinBox:focus {{ border-color: {ACCENT}; }}
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
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; border: 2px solid {ACCENT}; width: 14px; margin: -6px 0; border-radius: 9px; }}
QProgressBar {{ background: {BORDER}; border: none; border-radius: 6px; height: 12px; text-align: center;
    color: {TEXT}; font-weight: 600; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QGroupBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; margin-top: 16px; padding: 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; right: 12px; padding: 0 4px; color: {MUTED}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 10px; background: {CARD}; }}
QTabBar::tab {{ background: transparent; padding: 8px 14px; color: {MUTED}; font-weight: 600; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QStatusBar {{ background: {CARD}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QToolTip {{ background: {NAVY}; color: white; border: none; padding: 6px 8px; }}
"""


def _arrow_path() -> str:
    from core.paths import resource_root
    return (resource_root() / "app" / "assets" / "chevron_down.svg").as_posix()


QSS = QSS.replace("__ARROW__", _arrow_path())


def load_fonts() -> list[str]:
    """Register the bundled open-licensed fonts (app/assets/fonts, OFL). Needs a Qt application.
    Missing files are skipped: the stylesheet then falls back to Segoe UI."""
    from PySide6.QtGui import QFontDatabase

    from core.paths import resource_root
    families = []
    for f in sorted((resource_root() / "app" / "assets" / "fonts").glob("*.ttf")):
        i = QFontDatabase.addApplicationFont(str(f))
        families += QFontDatabase.applicationFontFamilies(i) if i >= 0 else []
    return sorted(set(families))


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(antialias=True, background=CARD, foreground=MUTED)
