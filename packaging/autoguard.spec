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
    (str(ROOT / "models" / "normal_behavior"), "models/normal_behavior"),
    (str(ROOT / "models" / "normal_behavior_drivetrain"), "models/normal_behavior_drivetrain"),
    (str(ROOT / "models" / "advanced"), "models/advanced"),
    (str(ROOT / "models" / "moment-1-small"), "models/moment-1-small"),
    (str(ROOT / "app" / "assets"), "app/assets"),
    (str(ROOT / "ATTRIBUTION.md"), "."),
]
# pint ships its unit registry as a text file (needed by python-OBD); arabic_reshaper has a default config.
datas += collect_data_files("pyqtgraph") + collect_data_files("pint") + collect_data_files("arabic_reshaper")
hiddenimports = (collect_submodules("momentfm") + collect_submodules("transformers.models.t5")
                 + collect_submodules("pyqtgraph") + collect_submodules("obd") + collect_submodules("bidi")
                 + collect_submodules("sklearn.ensemble._hist_gradient_boosting")
                 + ["sklearn.ensemble._iforest", "sklearn.linear_model._huber", "serial.tools.list_ports"])

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
               "IPython", "jupyter", "pytest"],
    noarchive=False,
)
# Qt uses Windows' unversioned ICU API. A third-party Poppler directory on PATH
# can make dependency discovery collect its incompatible version-suffixed ICU.
# Let Windows resolve its own icuuc.dll instead of shadowing it in _internal.
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() != "icuuc.dll"]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AutoGuardAI", console=False,
          icon=str(ROOT / "app" / "assets" / "app.ico"),
          upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AutoGuardAI")
