"""Application state shared by all pages: current session, replay engine,
analysis result and background workers. Pages talk to each other only
through these Qt signals."""
from __future__ import annotations

import logging
import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from ai.events import AnalysisResult
from ai.model_registry import ModelRegistry
from ai.pipeline import run_analysis
from ai.rules_baseline import RuleConfig
from core.playback.replay import ReplayEngine, ReplayState
from core.schema import NormalizedSession
from database.repositories import Repository

log = logging.getLogger(__name__)


class _WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    """Runs ``fn`` off the GUI thread and reports back through Qt signals."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            self.signals.finished.emit(self.fn(*self.args, **self.kwargs))
        except Exception as e:  # surfaced to the user as an actionable message
            log.error("worker failed: %s", traceback.format_exc())
            self.signals.failed.emit(f"{type(e).__name__}: {e}")


class AppState(QObject):
    session_changed = Signal()
    analysis_changed = Signal()
    replay_tick = Signal()             # playhead moved / rows appended
    replay_state_changed = Signal(str)
    busy_changed = Signal(bool, str)
    error = Signal(str)
    history_changed = Signal()
    focus_time = Signal(float, float)  # ask the Trip screen to show [t0, t1]

    TICK_MS = 100

    def __init__(self, repo: Repository):
        super().__init__()
        self.repo = repo
        self.session: NormalizedSession | None = None
        self.quality: dict | None = None
        self.session_id: int | None = None
        self.result: AnalysisResult | None = None
        self.findings: list = []
        self.run_id: int | None = None
        self.engine: ReplayEngine | None = None
        self.lang = repo.get_setting("language", "ar")
        self._generation = 0
        self._analysis_requests = {}
        self.pool = QThreadPool.globalInstance()
        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._on_tick)
        self.registry = self._make_registry()

    # ---- settings-derived objects -------------------------------------
    def _make_registry(self, vehicle_id: int | None = None) -> ModelRegistry:
        return ModelRegistry(enable_moment=bool(self.repo.get_setting("moment_enabled", True)),
                             moment_device=self.repo.get_setting("moment_device", "auto"), vehicle_id=vehicle_id,
                             powertrain=self.repo.vehicle_fuel(vehicle_id))

    def reload_registry(self) -> None:
        self.registry = self._make_registry(getattr(self.registry, "vehicle_id", None))

    def _registry_for_current(self) -> ModelRegistry:
        """The trip's car decides which expected-vs-actual model is used (personal or base)."""
        vid = self.repo.session_vehicle(self.session_id) if self.session_id is not None else None
        if getattr(self.registry, "vehicle_id", None) != vid:
            self.registry = self._make_registry(vid)
        return self.registry

    def rule_config(self) -> RuleConfig:
        saved = self.repo.get_setting("rule_config", {}) or {}
        cfg = RuleConfig()
        for k, v in saved.items():
            if hasattr(cfg, k):
                setattr(cfg, k, float(v))
        return cfg

    def vehicle_profile(self) -> dict | None:
        vid = self.repo.session_vehicle(self.session_id) if self.session_id is not None else None
        return next((v for v in self.repo.list_vehicles() if v["id"] == vid), None)

    # ---- session --------------------------------------------------------
    def load_source_async(self, source, then_analyse: bool = True) -> None:
        """Read + validate a VehicleDataSource off the GUI thread, save it, then analyse."""
        self.busy_changed.emit(True, "loading")
        self._generation += 1
        gen = self._generation
        vehicle_id = self.repo.get_setting("active_vehicle_id")

        def work():
            s = source.read_session()
            return gen, vehicle_id, then_analyse, s, (source.quality.to_dict() if getattr(source, "quality", None) else None)

        w = Worker(work)
        # Bound methods of this QObject => queued to the GUI thread (a plain closure
        # would run on the worker thread and touch widgets from there).
        w.signals.finished.connect(self._load_done)
        w.signals.failed.connect(self._load_failed)
        self.pool.start(w)

    def _load_done(self, res) -> None:
        gen, vehicle_id, then_analyse, s, q = res
        if gen != self._generation:
            return  # A newer import or an explicitly opened trip takes precedence.
        if q and q.get("status") == "Unusable":
            self.busy_changed.emit(False, "")
            self.error.emit("; ".join(q.get("warnings", [])) or "Unusable file")
            return
        if vehicle_id is not None and not any(v["id"] == vehicle_id for v in self.repo.list_vehicles()):
            vehicle_id = None  # The selected car may have been deleted while importing.
        sid = self.repo.save_session(s, q, vehicle_id)
        self.set_session(s, q, sid)
        self.history_changed.emit()
        self.busy_changed.emit(False, "")
        if then_analyse:
            self.run_analysis_async()

    def _load_failed(self, msg: str) -> None:
        self.busy_changed.emit(False, "")
        self.error.emit(msg)

    def set_session(self, session: NormalizedSession, quality: dict | None, session_id: int | None,
                    result: AnalysisResult | None = None, run_id: int | None = None) -> None:
        self.stop_replay()
        self._generation += 1          # invalidates analyses started for the previous trip
        self.session, self.quality, self.session_id = session, quality, session_id
        self.result, self.run_id = result, run_id
        self._refresh_findings()
        self.engine = ReplayEngine(session) if not session.data.empty else None
        self.session_changed.emit()
        self.analysis_changed.emit()
        self.replay_tick.emit()

    def open_saved_session(self, session_id: int) -> None:
        s, q = self.repo.load_session(session_id)
        latest = self.repo.load_latest_analysis(session_id)
        run_id, res = latest if latest else (None, None)
        self.set_session(s, q, session_id, res, run_id)

    # ---- replay ---------------------------------------------------------
    def play(self) -> None:
        if self.engine:
            self.engine.play()
            self._timer.start()
            self.replay_state_changed.emit(self.engine.state.value)

    def pause(self) -> None:
        if self.engine:
            self.engine.pause()
            self._timer.stop()
            self.replay_state_changed.emit(self.engine.state.value)

    def stop_replay(self) -> None:
        self._timer.stop()
        if self.engine:
            self.engine.stop()
            self.replay_state_changed.emit(self.engine.state.value)
            self.replay_tick.emit()

    def seek(self, elapsed_s: float) -> None:
        if self.engine:
            self.engine.seek(elapsed_s)
            self.replay_tick.emit()

    def set_speed(self, speed: float) -> None:
        if self.engine:
            self.engine.set_speed(speed)

    def _on_tick(self) -> None:
        if not self.engine:
            return
        rows = self.engine.tick()
        if len(rows) or self.engine.state == ReplayState.PLAYING:
            self.replay_tick.emit()
        if self.engine.state == ReplayState.FINISHED:
            self._timer.stop()
            self.replay_state_changed.emit(self.engine.state.value)

    # ---- analysis -------------------------------------------------------
    def run_analysis_async(self) -> None:
        if self.session is None:
            self.error.emit("Load a session first.")
            return
        self.busy_changed.emit(True, "analysing")
        self._generation += 1
        session, registry, cfg = self.session, self._registry_for_current(), self.rule_config()
        # Tag the job with the trip it belongs to: the user may open another trip
        # before it finishes, and the result must never be attached to that one.
        gen, sid = self._generation, self.session_id
        self._analysis_requests[sid] = gen
        w = Worker(lambda: (gen, sid, run_analysis(session, registry, cfg)))
        w.signals.finished.connect(self._analysis_done)
        w.signals.failed.connect(self._analysis_failed)
        self.pool.start(w)

    def _refresh_findings(self) -> None:
        from ai.explain import findings
        self.findings = findings(self.result, self.session) if self.session is not None else []

    def _analysis_done(self, tagged) -> None:
        gen, sid, result = tagged
        if self._analysis_requests.get(sid, gen) != gen:
            return  # A newer request for this trip supersedes this result in storage too.
        stale = gen != self._generation or sid != self.session_id
        exists = sid is not None and self.repo.conn.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone()
        try:
            run_id = self.repo.save_analysis(sid, result) if exists else None
        except Exception as exc:
            self._analysis_failed(str(exc))
            return
        if sid is not None and not exists:
            if not stale:
                self.busy_changed.emit(False, "")
            self.history_changed.emit()
            return
        if stale:
            self.history_changed.emit()
            return
        self.result, self.run_id = result, run_id
        self._refresh_findings()
        self.busy_changed.emit(False, f"done:{len([f for f in self.findings if f.confidence != 'low'])}")
        self.analysis_changed.emit()
        self.history_changed.emit()

    def _analysis_failed(self, msg: str) -> None:
        self.busy_changed.emit(False, "Analysis failed.")
        self.error.emit("Analysis failed: " + msg)
