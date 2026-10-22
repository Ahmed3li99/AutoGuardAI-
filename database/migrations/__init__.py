"""Ordered schema migrations. Append new (version, sql) tuples; never edit applied ones."""
from core.schema import NUMERIC_FIELDS

_READING_COLS = ",\n    ".join(f"{c} REAL" for c in NUMERIC_FIELDS)

MIGRATIONS: list[tuple[int, str]] = [
    (1, f"""
CREATE TABLE vehicles (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    make TEXT, model TEXT, year INTEGER, fuel TEXT, notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT, source_url TEXT, license TEXT, citation TEXT
);
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    source_name TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('recorded','synthetic','live_obd')),
    time_basis TEXT NOT NULL,
    dataset_id TEXT REFERENCES datasets(dataset_id),
    mapping_id TEXT,
    source_sha256 TEXT,
    vehicle_profile_id INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
    source_vehicle_id TEXT,
    source_trip_id TEXT,
    start_timestamp TEXT,
    playback_interval_s REAL,
    n_rows INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    signals_json TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    quality_json TEXT
);
CREATE TABLE readings (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    t_s REAL NOT NULL,
    {_READING_COLS},
    PRIMARY KEY (session_id, idx)
);
CREATE TABLE model_versions (
    version TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    description TEXT,
    artifact_sha256 TEXT,
    trained_on TEXT,
    threshold_json TEXT,
    registered_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE analysis_runs (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    detector_status_json TEXT NOT NULL,
    timing_json TEXT,
    config_json TEXT
);
CREATE TABLE anomaly_scores (
    run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    detector TEXT NOT NULL,
    t_json TEXT NOT NULL,
    score_json TEXT NOT NULL,
    threshold_json TEXT,
    PRIMARY KEY (run_id, detector)
);
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_start_s REAL NOT NULL, t_end_s REAL NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('attention','warning')),
    detector TEXT NOT NULL,
    detector_kind TEXT NOT NULL,
    signals_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    peak_score REAL, threshold REAL,
    model_version TEXT,
    values_json TEXT,
    origin TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_alerts_session ON alerts(session_id, t_start_s);
CREATE INDEX idx_runs_session ON analysis_runs(session_id);
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
"""),
]
