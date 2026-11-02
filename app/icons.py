"""Line icons drawn for AutoGuard AI (24x24 grid, round strokes), rendered from SVG at any colour/size.

Replaces emoji in the UI: emoji render differently on every Windows version and look informal.
"""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QLabel

from core.paths import resource_root

PATHS = {
    "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v10h13V10"/><path d="M10 20v-5.5h4V20"/>',
    "trip": '<circle cx="6" cy="18" r="2.2"/><circle cx="18" cy="6" r="2.2"/>'
            '<path d="M8.2 18H15a3 3 0 0 0 0-6H9a3 3 0 0 1 0-6h6.8"/>',
    "results": '<path d="M3 12.5h4l2.2-5.5 4.2 11 2.4-5.5H21"/>',
    "report": '<path d="M6 3h8.5L19 7.5V21H6z"/><path d="M14 3v5h5"/><path d="M9 12.5h6.5M9 16.5h6.5"/>',
    "settings": '<path d="M4 6.5h9M17.5 6.5H20M4 12h3M11.5 12H20M4 17.5h11M19 17.5h1"/>'
                '<circle cx="15.3" cy="6.5" r="2.2"/><circle cx="9.2" cy="12" r="2.2"/><circle cx="17" cy="17.5" r="2.2"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="3.8" ry="9"/><path d="M3.2 9h17.6M3.2 15h17.6"/>',
    "lock": '<rect x="5" y="10.5" width="14" height="10" rx="2.2"/><path d="M8.2 10.5V8a3.8 3.8 0 0 1 7.6 0v2.5"/>'
            '<path d="M12 14.5v2.5"/>',
    "folder": '<path d="M3 7.2A2.2 2.2 0 0 1 5.2 5h4l2.2 2.3h7.4A2.2 2.2 0 0 1 21 9.5v8.3a2.2 2.2 0 0 1-2.2 2.2H5.2'
              'A2.2 2.2 0 0 1 3 17.8z"/>',
    "car": '<path d="M4 15.5 5.6 10a2 2 0 0 1 1.9-1.4h9a2 2 0 0 1 1.9 1.4l1.6 5.5V19H4z"/><path d="M4 15.5h16"/>'
           '<circle cx="7.7" cy="19" r="1.6"/><circle cx="16.3" cy="19" r="1.6"/>',
    "plug": '<path d="M9 3v5M15 3v5"/><path d="M7 8h10v3.2a5 5 0 0 1-10 0z"/><path d="M12 16.2V21"/>',
    "flask": '<path d="M9 3h6M10 3v6.2l-5.2 8.9A2 2 0 0 0 6.5 21h11a2 2 0 0 0 1.7-2.9L14 9.2V3"/><path d="M7.4 15h9.2"/>',
    "sparkle": '<path d="M11 3.5 12.9 8.6 18 10.5 12.9 12.4 11 17.5 9.1 12.4 4 10.5 9.1 8.6z"/>'
               '<path d="M18.5 15.5 19.3 17.7 21.5 18.5 19.3 19.3 18.5 21.5 17.7 19.3 15.5 18.5 17.7 17.7z"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11"/><path d="M4.5 6h.01M4.5 12h.01M4.5 18h.01"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "alert": '<path d="M12 3.5 21.5 20h-19z"/><path d="M12 10v4.5M12 17.3h.01"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.4l2.7 2.7L16.2 9.6"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.8h.01"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5.2l3.3 2"/>',
    "download": '<path d="M12 4v11M7 10.5l5 5 5-5M5 20h14"/>',
    "play": '<path d="M8 5.5v13l10.5-6.5z"/>',
    "pause": '<path d="M9 5.5v13M15 5.5v13"/>',
    "stop": '<rect x="6.5" y="6.5" width="11" height="11" rx="1.5"/>',
    "shield": '<path d="M12 3 19 6v6c0 4.6-3 7.6-7 9-4-1.4-7-4.4-7-9V6z"/><path d="M9 12.2l2.2 2.2L15.2 10.4"/>',
    "gauge": '<path d="M4.5 17a8.5 8.5 0 1 1 15 0"/><path d="M12 13.5 16 9"/><circle cx="12" cy="14" r="1.4"/>',
    "trash": '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/>'
           '<path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/>',
}


def svg(name: str, color: str, stroke: float = 1.8) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{PATHS[name]}</svg>')


@lru_cache(maxsize=512)
def pixmap(name: str, color: str, size: int = 20, stroke: float = 1.8, ratio: float = 2.0) -> QPixmap:
    px = int(size * ratio)
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(QByteArray(svg(name, color, stroke).encode())).render(p, QRectF(0, 0, px, px))
    p.end()
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(ratio)
    return pm


def _padded(pm: QPixmap, size: int, pad: int, ratio: float = 2.0) -> QPixmap:
    """The icon centred in a wider transparent pixmap: a gap to the button text on either side (LTR and RTL)."""
    out = QImage(int((size + 2 * pad) * ratio), int(size * ratio), QImage.Format_ARGB32_Premultiplied)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.drawPixmap(QRectF(pad * ratio, 0, size * ratio, size * ratio), pm, QRectF(0, 0, pm.width(), pm.height()))
    p.end()
    res = QPixmap.fromImage(out)
    res.setDevicePixelRatio(ratio)
    return res


def icon(name: str, color: str, checked_color: str | None = None, size: int = 20, pad: int = 0) -> QIcon:
    ic = QIcon()
    for col, state in ((color, QIcon.Off), (checked_color or color, QIcon.On)):
        pm = pixmap(name, col, size)
        ic.addPixmap(_padded(pm, size, pad) if pad else pm, QIcon.Normal, state)
    return ic


def tile(name: str, fg: str, bg: str, size: int = 44, radius: int = 12) -> QLabel:
    """A rounded square with an icon in it (used on cards instead of emoji)."""
    lab = QLabel()
    lab.setPixmap(pixmap(name, fg, int(size * 0.5)))
    lab.setAlignment(Qt.AlignCenter)
    lab.setFixedSize(size, size)
    lab.setStyleSheet(f"background:{bg}; border-radius:{radius}px;")
    return lab


def logo_path(size: int = 64) -> str:
    return str(resource_root() / "app" / "assets" / f"logo_{size}.png")


def pixmap_file(path: str, size: int, ratio: float = 2.0) -> QPixmap:
    pm = QPixmap(path).scaled(int(size * ratio), int(size * ratio), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    pm.setDevicePixelRatio(ratio)
    return pm


def app_icon() -> QIcon:
    ic = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        ic.addFile(logo_path(s))
    return ic
