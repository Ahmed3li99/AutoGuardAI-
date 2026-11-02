"""Regressions for reference integrity, stale jobs and scenario comparability."""
import numpy as np
import pandas as pd
import pytest

from ai.events import Alert, AnalysisResult
from core.schema import NormalizedSession, SessionMetadata, DataOrigin, TimeBasis
from database.connection import connect
from database.repositories import Repository


def session():
    return NormalizedSession(pd.DataFrame({"t_s": np.arange(120.), "rpm": 1400.,
                                          "speed_kmh": 40., "coolant_temp_c": 90.}),
                             SessionMetadata("reference", DataOrigin.RECORDED, TimeBasis.RELATIVE_RECORDED))


def result(alerts=None):
    return AnalysisResult(alerts or [], {}, {"rules": {"status": "ok"}}, {}, {})


@pytest.mark.parametrize("kind", ["heuristic_rule", "data_consistency"])
def test_reanalysis_revokes_reference_and_reapproval_rejected(tmp_path, kind):
    repo = Repository(connect(tmp_path / "refs.db"))
    vid = repo.save_vehicle("A")
    sid = repo.save_session(session(), vehicle_profile_id=vid)
    repo.save_analysis(sid, result()); repo.set_reference(sid, True)
    assert repo.reference_sessions(vid) == [sid]
    repo.save_analysis(sid, result([Alert(10, 20, "warning", "test", kind, ["rpm"], "bad")]))
    assert not repo.reference_sessions(vid)
    with pytest.raises(ValueError): repo.set_reference(sid, True)


def test_changing_car_revokes_reference(tmp_path):
    repo = Repository(connect(tmp_path / "refs.db"))
    a, b = repo.save_vehicle("A"), repo.save_vehicle("B")
    sid = repo.save_session(session(), vehicle_profile_id=a)
    repo.save_analysis(sid, result()); repo.set_reference(sid, True)
    repo.set_session_vehicle(sid, a)
    assert repo.reference_sessions(a) == [sid]
    repo.set_session_vehicle(sid, b)
    assert not repo.reference_sessions(a) and not repo.reference_sessions(b)


def test_long_dropout_detected_but_leading_absence_not_dropout():
    from ai.consistency import run_consistency
    d = session().data
    d.loc[20:, "coolant_temp_c"] = np.nan
    assert any("dropout" in a.detector and "coolant_temp_c" in a.signals for a in run_consistency(d))
    d = session().data; d.loc[:100, "coolant_temp_c"] = np.nan
    assert not any("dropout" in a.detector for a in run_consistency(d))


def test_deleted_trip_completion_and_superseded_job(tmp_path):
    from PySide6.QtWidgets import QApplication
    from app.state import AppState
    _app = QApplication.instance() or QApplication([])  # keep a Qt app alive for widgets
    repo = Repository(connect(tmp_path / "jobs.db")); st = AppState(repo)
    s = session(); sid = repo.save_session(s)
    st.set_session(s, None, sid)
    old, new = st._generation, st._generation+1
    st._generation = new; st._analysis_requests[sid] = new
    st._analysis_done((new, sid, result()))
    latest = repo.load_latest_analysis(sid)[0]
    st._analysis_done((old, sid, result()))
    assert repo.load_latest_analysis(sid)[0] == latest
    repo.delete_session(sid)
    errors = []; st.error.connect(errors.append)
    st._analysis_done((new, sid, result()))
    assert not errors and repo.load_latest_analysis(sid) is None


def test_report_uses_trip_car_not_active_selection(tmp_path):
    from PySide6.QtWidgets import QApplication
    from app.state import AppState
    _app = QApplication.instance() or QApplication([])  # keep a Qt app alive for widgets
    repo = Repository(connect(tmp_path / "cars.db"))
    a, b = repo.save_vehicle("Trip car"), repo.save_vehicle("Selected car")
    s = session(); sid = repo.save_session(s, vehicle_profile_id=a)
    repo.set_setting("active_vehicle_id", b)
    st = AppState(repo); st.set_session(s, None, sid)
    assert st.vehicle_profile()["id"] == a


def test_scenario_recomputes_before_under_current_configuration(monkeypatch):
    import ai.scenarios as mod
    calls = []
    def analyse(s, registry, cfg):
        calls.append((s, cfg)); return result()
    monkeypatch.setattr(mod, "run_analysis", analyse)
    stored = result([Alert(0, 10, "warning", "rule:old", "heuristic_rule", ["rpm"], "old")])
    cfg = object()
    compared = mod.run_scenario(session(), [], object(), cfg, before=stored)
    assert len(calls) == 2 and all(c[1] is cfg for c in calls)
    assert not compared.result_before.alerts and not compared.diff["new"]


def test_what_if_selects_model_for_available_signals(monkeypatch):
    import ai.scenarios as mod
    from types import SimpleNamespace
    captured = []
    s = session()
    model = SimpleNamespace(models={"rpm": True},
                            predict=lambda d: pd.DataFrame({"rpm__expected": np.full(len(d), 1300.)}))
    def choose(signals):
        captured.append(signals); return model
    monkeypatch.setattr(mod, "run_analysis", lambda *args: result())
    mod.what_if_normal(s, "rpm", 20, 30, SimpleNamespace(expected_vs_actual=choose))
    assert captured == [s.signals]


def test_import_keeps_captured_car_and_discards_superseded_load(tmp_path):
    from PySide6.QtWidgets import QApplication
    from app.state import AppState
    _app = QApplication.instance() or QApplication([])  # keep a Qt app alive for widgets
    repo = Repository(connect(tmp_path / "imports.db"))
    a, b = repo.save_vehicle("A"), repo.save_vehicle("B")
    state = AppState(repo); state._generation = 2
    repo.set_setting("active_vehicle_id", b)
    state._load_done((1, b, False, session(), None))
    assert repo.list_sessions().empty
    state._load_done((2, a, False, session(), None))
    assert repo.session_vehicle(state.session_id) == a


def test_live_recording_waits_for_worker_before_import(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QApplication
    from app.state import AppState
    from app.pages.live_dialog import LiveDialog
    _app = QApplication.instance() or QApplication([])  # keep a Qt app alive for widgets
    state = AppState(Repository(connect(tmp_path / "live.db")))
    loaded = []; stopped = []
    monkeypatch.setattr(state, "load_source_async", loaded.append)
    dialog = LiveDialog(state)
    dialog.source = SimpleNamespace(recorded=[None]*5)
    dialog.worker = SimpleNamespace(stop=lambda: stopped.append(True))
    dialog.recording = True
    dialog.toggle_record()
    assert stopped and not loaded
    dialog._stopped()
    assert len(loaded) == 1
    dialog.close()


@pytest.mark.parametrize("ending",["disconnect","escape"])
def test_live_disconnect_or_escape_keeps_partial_recording(tmp_path, monkeypatch, ending):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QApplication
    from app.state import AppState
    from app.pages.live_dialog import LiveDialog
    _app = QApplication.instance() or QApplication([])  # keep a Qt app alive for widgets
    state=AppState(Repository(connect(tmp_path/"partial.db")))
    loaded=[]; stopped=[]
    monkeypatch.setattr(state,"load_source_async",loaded.append)
    dialog=LiveDialog(state); dialog.source=SimpleNamespace(recorded=[None]*8)
    dialog.worker=SimpleNamespace(stop=lambda:stopped.append(True)); dialog.recording=True
    if ending=="escape":
        dialog.reject(); assert stopped and not loaded
    dialog._stopped()
    assert not dialog.recording and len(loaded)==1
    dialog.close()
