"""Application entry point."""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logging() -> None:
    from core.paths import user_data_dir
    log_dir = user_data_dir() / "logs"
    log_dir.mkdir(exist_ok=True)
    handler = RotatingFileHandler(log_dir / "autoguard.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def selftest(out_dir: str) -> int:
    """Headless end-to-end check of the (frozen) app: bundled sample -> validate ->
    all detectors -> SQLite -> PDF/CSV. Writes selftest.json into out_dir."""
    import json
    import time
    from pathlib import Path

    from ai.model_registry import ModelRegistry
    from ai.pipeline import overall_status, run_analysis
    from core.data_sources.files import RecordedFileSource
    from core.paths import SAMPLE_DIR, resource_root
    from database.connection import connect
    from database.repositories import Repository
    from reporting.csv_export import export_alerts_csv
    from reporting.pdf_export import build_pdf_report

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"frozen": bool(getattr(sys, "frozen", False)), "resource_root": str(resource_root())}
    try:
        t0 = time.perf_counter()
        src = RecordedFileSource(SAMPLE_DIR / "toyota_carobd_long10.csv")
        s = src.read_session()
        repo = Repository(connect(out / "selftest.sqlite3"))
        sid = repo.save_session(s, src.quality.to_dict())
        r = run_analysis(s, ModelRegistry())
        repo.save_analysis(sid, r)
        s2, q2 = repo.load_session(sid)
        _, r2 = repo.load_latest_analysis(sid)
        build_pdf_report(out / "selftest_report.pdf", s2, q2, r2, sid)
        export_alerts_csv(s2, r2, out / "selftest_alerts.csv", sid)
        res.update(ok=True, rows=len(s.data), quality=src.quality.status, n_alerts=len(r.alerts),
                   status=overall_status(r, s), detectors=r.detector_status,
                   timing_s=r.timing_s, total_s=round(time.perf_counter() - t0, 2),
                   persisted_alerts=len(r2.alerts))
    except Exception as e:  # report, don't crash silently
        import traceback
        res.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    (out / "selftest.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return 0 if res.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    # Never reach out to the Hugging Face Hub at runtime.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    setup_logging()
    args = argv if argv is not None else sys.argv
    if "--selftest" in args:
        i = args.index("--selftest")
        return selftest(args[i + 1] if len(args) > i + 1 else ".")
    from PySide6.QtWidgets import QApplication

    from app.main_window import MainWindow
    from app.state import AppState
    from app.theme import QSS, configure_pyqtgraph
    from database.connection import connect
    from database.repositories import Repository

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("AutoGuard AI")
    app.setStyleSheet(QSS)
    configure_pyqtgraph()
    state = AppState(Repository(connect()))
    win = MainWindow(state)
    win.show()
    if os.environ.get("AUTOGUARD_SMOKE_TEST") == "1":   # used by tests / packaging check
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1500, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
