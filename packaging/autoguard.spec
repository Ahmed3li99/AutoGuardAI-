# PyInstaller spec - onedir build.  Run from the project root:
#   .venv\Scripts\pyinstaller packaging\autoguard.spec --noconfirm
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

datas = [
    (str(ROOT / "datasets" / "column_mappings"), "datasets/column_mappings"),
    (str(ROOT / "datasets" / "datasets_manifest.json"), "datasets"),
    (str(ROOT / "data" / "sample"), "data/sample"),
    (str(ROOT / "models" / "iforest"), "models/iforest"),
    (str(ROOT / "models" / "moment-1-small"), "models/moment-1-small"),
    (str(ROOT / "ATTRIBUTION.md"), "."),
]
datas += collect_data_files("pyqtgraph")
hiddenimports = (collect_submodules("momentfm") + collect_submodules("transformers.models.t5")
                 + collect_submodules("pyqtgraph") + ["sklearn.ensemble._iforest"])

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
               "IPython", "jupyter", "pytest", "obd"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AutoGuardAI", console=False,
          upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AutoGuardAI")
