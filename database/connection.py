"""SQLite connection with schema versioning."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from core.paths import user_data_dir
from database.migrations import MIGRATIONS


def default_db_path() -> Path:
    return user_data_dir() / "autoguard.sqlite3"


def schema_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL, "
                 "applied_at TEXT NOT NULL DEFAULT (datetime('now')))")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def migrate(conn: sqlite3.Connection) -> int:
    current = schema_version(conn)
    for version, sql in MIGRATIONS:
        if version > current:
            with conn:
                conn.executescript(sql)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            current = version
    return current


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(path) if path is not None else default_db_path()
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn
