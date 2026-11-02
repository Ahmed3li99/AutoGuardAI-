"""Render the AutoGuard AI logo (app/assets/logo.svg) into the raster assets the app and the EXE need.

Usage:  python tools/make_brand_assets.py
Output: app/assets/logo_<n>.png (16-512 px) and app/assets/app.ico (multi-size, PNG-compressed entries)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app" / "assets"
SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256, 512]
ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]


def render(svg: Path, size: int) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    QSvgRenderer(str(svg)).render(p)
    p.end()
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


def write_ico(path: Path, pngs: dict[int, bytes]) -> None:
    """ICO container with PNG entries (supported since Windows Vista)."""
    sizes = sorted(pngs)
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries, blobs = b"", b""
    for s in sizes:
        data = pngs[s]
        dim = 0 if s >= 256 else s
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset + len(blobs))
        blobs += data
    path.write_bytes(header + entries + blobs)


def main() -> int:
    from PySide6.QtGui import QGuiApplication
    _app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    svg = ASSETS / "logo.svg"
    pngs = {s: render(svg, s) for s in SIZES}
    for s, data in pngs.items():
        (ASSETS / f"logo_{s}.png").write_bytes(data)
    write_ico(ASSETS / "app.ico", {s: pngs[s] for s in ICO_SIZES})
    print("wrote", len(pngs), "PNGs and app.ico to", ASSETS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
