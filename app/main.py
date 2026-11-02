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
        build_pdf_report(out / "selftest_report_en.pdf", s2, q2, r2, sid, lang="en")
        build_pdf_report(out / "selftest_report_ar.pdf", s2, q2, r2, sid, lang="ar")
        export_alerts_csv(s2, r2, out / "selftest_alerts.csv", sid)
        from ai.explain import findings, verdict
        fs = findings(r2, s2)
        # demo with injected faults must be found and explained
        from core.data_sources.synthetic_demo import SyntheticDemoDataSource
        demo = SyntheticDemoDataSource().read_session()
        dfs = findings(run_analysis(demo, ModelRegistry()), demo)
        # simulated adapter through the live OBD code path (not a hardware test)
        from core.data_sources.obd_optional import OBDDataSource
        from core.data_sources.obd_simulator import simulated_obd_module
        sim = OBDDataSource(None, enabled=True, obd_module=simulated_obd_module(), simulated=True)
        sim.open()
        sim_vals = sim.poll_once().values
        sim.close()
        # v0.3 features inside the frozen build: scenario lab, what-if, car adaptation
        from ai.normal_behavior import NormalBehaviorModel
        from ai.personal import adapt_base_model
        from ai.scenarios import run_scenario, what_if_normal
        from core.analytics.injection import InjectedEvent
        from core.preprocessing.resample import resample_1hz
        from core.schema import DataOrigin, NormalizedSession, SessionMetadata, TimeBasis
        lab_reg = ModelRegistry(enable_moment=False)
        lab = run_scenario(s, [InjectedEvent("voltage_decline", "battery_voltage", 1000, 1600, "selftest", 11.6)], lab_reg)
        wf = what_if_normal(demo, "battery_voltage", 1150, 1350, lab_reg)
        frames = [resample_1hz(x.data, x.signals)[0] for x in (s, RecordedFileSource(SAMPLE_DIR / "toyota_carobd_ufpe7.csv").read_session())]
        for f in frames:                    # a "hotter" car, so the adaptation code path really runs
            f["coolant_temp_c"] = f["coolant_temp_c"] + 6.0
        base_nbm = lab_reg.expected_vs_actual()
        adapted = adapt_base_model(base_nbm, frames[:1], frames[1:], {"version": "selftest"})
        # other cars: an RPM + speed-only log uses the drivetrain model; a hybrid profile skips "0 RPM moving"
        dt_df = frames[0][["t_s", "rpm", "speed_kmh"]].copy()
        dt_df.loc[600:640, "rpm"] = 0.0
        dt_sess = NormalizedSession(dt_df, SessionMetadata(source_name="rpm_speed_only", origin=DataOrigin.RECORDED,
                                                           time_basis=TimeBasis.RELATIVE_RECORDED))
        dt_r = run_analysis(dt_sess, ModelRegistry(enable_moment=False, powertrain="hybrid"))
        res.update(drivetrain_variant=dt_r.detector_status["expected_vs_actual"].get("variant"),
                   hybrid_rpm0_alerts=sum(a.detector == "consistency:moving_rpm0" for a in dt_r.alerts))
        assert res["drivetrain_variant"] == "drivetrain" and res["hybrid_rpm0_alerts"] == 0
        from ai.consistency import run_consistency
        glitch = frames[0].copy()
        glitch.loc[300, "rpm"] = 11596.0             # real Ford-log glitch value
        from PySide6.QtGui import QGuiApplication
        if QGuiApplication.instance() is None:          # pixmaps need a Qt app; offscreen = no window
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            _qt = QGuiApplication(["selftest"])  # noqa: F841
        from app.icons import app_icon, logo_path, pixmap
        res["brand_assets"] = {"logo": Path(logo_path(64)).exists(), "app_icon_sizes": len(app_icon().availableSizes()),
                               "line_icon": not pixmap("home", "#000000", 20).isNull()}
        assert all(res["brand_assets"].values())
        res["spike_check"] = sorted({a.detector for a in run_consistency(glitch) if a.detector.endswith("_spike")})
        assert res["spike_check"] == ["consistency:rpm_spike"]
        res.update(lab_detection={d: v["delay_s"] for d, v in lab.per_detector.items()},
                   what_if_gone=[f.en["title"] for f in wf.diff["gone"] if f.confidence != "low"],
                   adaptation_car_map=adapted.meta.get("car_map"),
                   consistency_status=r.detector_status.get("consistency", {}).get("status"))
        assert isinstance(base_nbm, NormalBehaviorModel)
        res.update(ok=True, rows=len(s.data), quality=src.quality.status, n_alerts=len(r.alerts),
                   status=overall_status(r, s), verdict=verdict(fs, r2, s2)["en"],
                   findings=[f.en["title"] for f in fs if f.confidence != "low"],
                   demo_findings=[f"{f.severity}: {f.en['title']}" for f in dfs if f.confidence != "low"],
                   simulated_adapter_signals=sorted(sim_vals), detectors=r.detector_status,
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
    from app.theme import load_fonts
    load_fonts()
    app.setStyleSheet(QSS)
    from app.icons import app_icon
    app.setWindowIcon(app_icon())
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
