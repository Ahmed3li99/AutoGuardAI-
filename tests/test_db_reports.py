"""SQLite migrations/history, alert persistence & provenance, report consistency."""
import sqlite3

import pandas as pd
import pytest

from ai.model_registry import ModelRegistry
from ai.pipeline import run_analysis
from core.data_sources.files import RecordedFileSource
from core.data_sources.synthetic_demo import SyntheticDemoDataSource
from database.connection import connect, schema_version
from database.migrations import MIGRATIONS
from database.repositories import Repository
from reporting.csv_export import alerts_frame, export_alerts_csv
from reporting.pdf_export import build_pdf_report


@pytest.fixture
def repo(tmp_path):
    return Repository(connect(tmp_path / "t.sqlite3"))


def test_migrations_idempotent(tmp_path):
    p = tmp_path / "m.sqlite3"
    c1 = connect(p)
    assert schema_version(c1) == MIGRATIONS[-1][0]
    c1.close()
    c2 = connect(p)                                   # re-open: no re-application
    assert c2.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == len(MIGRATIONS)


def test_session_roundtrip_and_cascade_delete(repo, toyota_sample):
    src = RecordedFileSource(toyota_sample)
    s = src.read_session()
    sid = repo.save_session(s, src.quality.to_dict())
    s2, q2 = repo.load_session(sid)
    pd.testing.assert_frame_equal(s.data[s2.data.columns].astype(float), s2.data.astype(float))
    assert s2.meta.source_sha256 == s.meta.source_sha256 and q2["rows"] == len(s.data)
    r = run_analysis(s, ModelRegistry(enable_moment=False))
    repo.save_analysis(sid, r)
    repo.delete_session(sid)
    for t in ("readings", "analysis_runs", "alerts", "anomaly_scores"):
        assert repo.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0


def test_alert_persistence_and_provenance(repo):
    s = SyntheticDemoDataSource().read_session()
    r = run_analysis(s, ModelRegistry(enable_moment=False))
    sid = repo.save_session(s)
    repo.save_analysis(sid, r)
    _, r2 = repo.load_latest_analysis(sid)
    assert [a.to_dict() for a in r2.alerts] == [a.to_dict() for a in r.alerts]
    df = repo.list_alerts(sid)
    assert set(df["origin"]) == {"synthetic"}
    assert df["model_version"].notna().all()
    assert repo.list_sessions().iloc[0]["n_alerts"] == len(r.alerts)


def test_origin_constraint(repo):
    with pytest.raises(sqlite3.IntegrityError):
        with repo.conn:
            repo.conn.execute("INSERT INTO sessions (source_name, origin, time_basis, n_rows, duration_s, "
                              "signals_json, meta_json) VALUES ('x','made_up','x',0,0,'[]','{}')")


def test_delete_all_history(repo):
    s = SyntheticDemoDataSource(duration_s=100).read_session()
    repo.save_session(s)
    repo.save_session(s)
    repo.delete_all_history()
    assert repo.list_sessions().empty


def test_reports_match_persisted_results(repo, tmp_path):
    """Charts/alerts/report consistency for a small deterministic sample."""
    src = SyntheticDemoDataSource()
    s = src.read_session()
    r = run_analysis(s, ModelRegistry(enable_moment=False))
    sid = repo.save_session(s, src.quality.to_dict())
    repo.save_analysis(sid, r)
    s2, q2 = repo.load_session(sid)
    _, r2 = repo.load_latest_analysis(sid)
    csv = pd.read_csv(export_alerts_csv(s2, r2, tmp_path / "a.csv", sid))
    db = repo.list_alerts(sid)
    assert len(csv) == len(db) == len(r.alerts)
    assert csv["t_start_s"].tolist() == db["t_start_s"].tolist()
    assert (csv["data_origin"] == "synthetic").all()
    pdf = build_pdf_report(tmp_path / "r.pdf", s2, q2, r2, sid)
    assert pdf.stat().st_size > 10_000
    text = _pdf_text(pdf)
    assert "SYNTHETIC DEMO DATA" in text
    assert f"Alert timeline ({len(r.alerts)})" in text


def test_report_without_analysis_omits_results(tmp_path, toyota_sample):
    src = RecordedFileSource(toyota_sample)
    s = src.read_session()
    text = _pdf_text(build_pdf_report(tmp_path / "r.pdf", s, src.quality.to_dict(), None))
    assert "No analysis has been run" in text and "SYNTHETIC" not in text
    assert alerts_frame(s, run_analysis(s, ModelRegistry(enable_moment=False))).columns[0] == "session_id"


def _pdf_text(p):
    try:
        import pymupdf
    except ImportError:
        pytest.skip("pymupdf not installed (dev-only dependency)")
    return "\n".join(page.get_text() for page in pymupdf.open(p))
