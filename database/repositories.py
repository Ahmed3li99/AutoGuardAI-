"""Data access for sessions, readings, analysis runs, alerts and settings."""
from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from ai.events import Alert, AnalysisResult, DetectorScores
from core.schema import NUMERIC_FIELDS, TIME_COL, NormalizedSession, SessionMetadata


def _j(x: Any) -> str:
    return json.dumps(x, default=str)


def _clean_float(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


class Repository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---- datasets / vehicles -----------------------------------------
    def upsert_dataset(self, dataset_id: str, name: str = "", source_url: str = "",
                       license: str = "", citation: str = "") -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO datasets (dataset_id, name, source_url, license, citation) VALUES (?,?,?,?,?) "
                "ON CONFLICT(dataset_id) DO UPDATE SET name=excluded.name, source_url=excluded.source_url, "
                "license=excluded.license, citation=excluded.citation",
                (dataset_id, name, source_url, license, citation))

    def save_vehicle(self, name: str, make: str = "", model: str = "", year: int | None = None,
                     fuel: str = "", notes: str = "", vehicle_id: int | None = None) -> int:
        with self.conn:
            if vehicle_id is None:
                cur = self.conn.execute(
                    "INSERT INTO vehicles (name, make, model, year, fuel, notes) VALUES (?,?,?,?,?,?)",
                    (name, make, model, year, fuel, notes))
                return int(cur.lastrowid)
            self.conn.execute("UPDATE vehicles SET name=?, make=?, model=?, year=?, fuel=?, notes=? WHERE id=?",
                              (name, make, model, year, fuel, notes, vehicle_id))
            return vehicle_id

    def set_vehicle_fuel(self, vehicle_id: int, fuel: str) -> None:
        """Powertrain of the car profile: "", "ice", "hybrid" or "ev"."""
        with self.conn:
            self.conn.execute("UPDATE vehicles SET fuel=? WHERE id=?", (fuel, vehicle_id))

    def vehicle_fuel(self, vehicle_id: int | None) -> str:
        if vehicle_id is None:
            return ""
        r = self.conn.execute("SELECT fuel FROM vehicles WHERE id=?", (vehicle_id,)).fetchone()
        return (r[0] or "") if r else ""

    def list_vehicles(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM vehicles ORDER BY id")]

    # ---- sessions -----------------------------------------------------
    def save_session(self, session: NormalizedSession, quality: dict | None = None,
                     vehicle_profile_id: int | None = None) -> int:
        m = session.meta
        df = session.data
        if m.dataset_id:
            exists = self.conn.execute("SELECT 1 FROM datasets WHERE dataset_id=?", (m.dataset_id,)).fetchone()
            if not exists:
                self.upsert_dataset(m.dataset_id, m.dataset_id)
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO sessions (source_name, origin, time_basis, dataset_id, mapping_id, source_sha256, "
                "vehicle_profile_id, source_vehicle_id, source_trip_id, start_timestamp, playback_interval_s, "
                "n_rows, duration_s, signals_json, meta_json, quality_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (m.source_name, m.origin.value, m.time_basis.value, m.dataset_id, m.mapping_id, m.source_sha256,
                 vehicle_profile_id, m.vehicle_id, m.trip_id, m.start_timestamp, m.playback_interval_s,
                 len(df), session.duration_s, _j(session.signals), _j(m.to_dict()),
                 _j(quality) if quality is not None else None))
            sid = int(cur.lastrowid)
            cols = [c for c in NUMERIC_FIELDS if c in df.columns]
            sub = df[[TIME_COL, *cols]].astype(float)
            arr = sub.to_numpy()
            rows = [(sid, i, *[None if np.isnan(v) else float(v) for v in arr[i]]) for i in range(len(arr))]
            ph = ",".join("?" * (2 + 1 + len(cols)))
            self.conn.executemany(f"INSERT INTO readings (session_id, idx, t_s, {', '.join(cols)}) VALUES ({ph})"
                                  if cols else f"INSERT INTO readings (session_id, idx, t_s) VALUES ({ph})", rows)
        return sid

    def list_sessions(self, vehicle_id: int | None = None) -> pd.DataFrame:
        latest = "(SELECT MAX(id) FROM analysis_runs r WHERE r.session_id = s.id)"
        q = f"""SELECT s.id, s.created_at, s.source_name, s.origin, s.time_basis, s.dataset_id, s.n_rows,
                      s.duration_s, s.signals_json, s.vehicle_profile_id, s.is_reference,
                      (SELECT COUNT(*) FROM alerts a WHERE a.session_id = s.id AND a.run_id = {latest}) AS n_alerts,
                      (SELECT COUNT(*) FROM alerts a WHERE a.session_id = s.id AND a.run_id = {latest}
                         AND a.detector_kind = 'heuristic_rule' AND a.severity = 'warning') AS n_rule_warnings,
                      {latest} AS last_run_id
               FROM sessions s {"WHERE s.vehicle_profile_id = ?" if vehicle_id is not None else ""}
               ORDER BY s.id DESC"""
        return pd.read_sql_query(q, self.conn, params=(vehicle_id,) if vehicle_id is not None else ())

    # ---- "learn your car" -------------------------------------------------
    def session_vehicle(self, session_id: int) -> int | None:
        r = self.conn.execute("SELECT vehicle_profile_id FROM sessions WHERE id=?", (session_id,)).fetchone()
        return r[0] if r else None

    def set_session_vehicle(self, session_id: int, vehicle_id: int | None) -> None:
        with self.conn:
            self.conn.execute("UPDATE sessions SET is_reference=CASE WHEN vehicle_profile_id IS ? THEN is_reference "
                              "ELSE 0 END, vehicle_profile_id=? WHERE id=?", (vehicle_id, vehicle_id, session_id))

    def set_reference(self, session_id: int, is_reference: bool) -> None:
        """Mark a trip as a user-approved normal reference trip. Refused for trips whose
        latest analysis contains a rule warning (the app must not learn a fault as normal)."""
        if is_reference:
            row = self.list_sessions().set_index("id").loc[session_id]
            if pd.isna(row["last_run_id"]):
                raise ValueError("Analyse the trip before using it as a reference.")
            if int(row["n_rule_warnings"]) > 0:
                raise ValueError("This trip has a confirmed warning, so it cannot be a normal reference trip.")
            if self.conn.execute("SELECT 1 FROM alerts WHERE run_id=? AND detector_kind='data_consistency' LIMIT 1",
                                 (int(row["last_run_id"]),)).fetchone():
                raise ValueError("Resolve data consistency problems before approving a reference trip.")
            if pd.isna(row["vehicle_profile_id"]):
                raise ValueError("Assign the trip to a car first.")
        with self.conn:
            self.conn.execute("UPDATE sessions SET is_reference=? WHERE id=?", (int(is_reference), session_id))

    def reference_sessions(self, vehicle_id: int) -> list[int]:
        return [r[0] for r in self.conn.execute(
            "SELECT s.id FROM sessions s WHERE vehicle_profile_id=? AND is_reference=1 "
            "AND EXISTS (SELECT 1 FROM analysis_runs r WHERE r.session_id=s.id) "
            "AND NOT EXISTS (SELECT 1 FROM alerts a WHERE a.run_id=(SELECT MAX(id) FROM analysis_runs r "
            "WHERE r.session_id=s.id) AND ((a.detector_kind='heuristic_rule' AND a.severity='warning') "
            "OR a.detector_kind='data_consistency')) ORDER BY s.id", (vehicle_id,))]

    def set_vehicle_model(self, vehicle_id: int, info: dict | None) -> None:
        with self.conn:
            self.conn.execute("UPDATE vehicles SET model_json=? WHERE id=?",
                              (_j(info) if info is not None else None, vehicle_id))

    def vehicle_model(self, vehicle_id: int) -> dict | None:
        r = self.conn.execute("SELECT model_json FROM vehicles WHERE id=?", (vehicle_id,)).fetchone()
        return json.loads(r[0]) if r and r[0] else None

    def delete_vehicle(self, vehicle_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE sessions SET is_reference=0 WHERE vehicle_profile_id=?", (vehicle_id,))
            self.conn.execute("DELETE FROM vehicles WHERE id=?", (vehicle_id,))

    def load_session(self, session_id: int) -> tuple[NormalizedSession, dict | None]:
        row = self.conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"No session {session_id}")
        meta = SessionMetadata.from_dict(json.loads(row["meta_json"]))
        df = pd.read_sql_query("SELECT * FROM readings WHERE session_id=? ORDER BY idx", self.conn,
                               params=(session_id,))
        df = df.drop(columns=["session_id", "idx"]).dropna(axis=1, how="all")
        if TIME_COL not in df:
            df[TIME_COL] = pd.Series(dtype=float)
        quality = json.loads(row["quality_json"]) if row["quality_json"] else None
        return NormalizedSession(df.reset_index(drop=True), meta), quality

    def delete_session(self, session_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def delete_all_history(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM sessions")
        self.conn.execute("VACUUM")

    # ---- models / analysis ---------------------------------------------
    def register_model(self, version: str, kind: str, description: str = "", artifact_sha256: str = "",
                       trained_on: str = "", threshold: dict | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO model_versions (version, kind, description, artifact_sha256, trained_on, "
                "threshold_json) VALUES (?,?,?,?,?,?)",
                (version, kind, description, artifact_sha256, trained_on, _j(threshold)))

    def save_analysis(self, session_id: int, result: AnalysisResult) -> int:
        with self.conn:
            if any((a.detector_kind == "heuristic_rule" and a.severity == "warning")
                   or a.detector_kind == "data_consistency" for a in result.alerts):
                self.conn.execute("UPDATE sessions SET is_reference=0 WHERE id=?", (session_id,))
            cur = self.conn.execute(
                "INSERT INTO analysis_runs (session_id, detector_status_json, timing_json, config_json) VALUES (?,?,?,?)",
                (session_id, _j(result.detector_status), _j(result.timing_s), _j(result.config)))
            run_id = int(cur.lastrowid)
            for det, sc in result.scores.items():
                self.conn.execute(
                    "INSERT INTO anomaly_scores (run_id, detector, t_json, score_json, threshold_json) VALUES (?,?,?,?,?)",
                    (run_id, det, _j(sc.t_s), _j([_clean_float(s) for s in sc.scores]), _j(sc.threshold)))
            self.conn.executemany(
                "INSERT INTO alerts (run_id, session_id, t_start_s, t_end_s, severity, detector, detector_kind, "
                "signals_json, explanation, peak_score, threshold, model_version, values_json, origin) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(run_id, session_id, a.t_start_s, a.t_end_s, a.severity, a.detector, a.detector_kind,
                  _j(a.signals), a.explanation, _clean_float(a.peak_score), _clean_float(a.threshold),
                  a.model_version, _j(a.values), a.origin) for a in result.alerts])
        return run_id

    def load_latest_analysis(self, session_id: int) -> tuple[int, AnalysisResult] | None:
        run = self.conn.execute("SELECT * FROM analysis_runs WHERE session_id=? ORDER BY id DESC LIMIT 1",
                                (session_id,)).fetchone()
        if run is None:
            return None
        scores = {}
        for r in self.conn.execute("SELECT * FROM anomaly_scores WHERE run_id=?", (run["id"],)):
            scores[r["detector"]] = DetectorScores(json.loads(r["t_json"]), json.loads(r["score_json"]),
                                                   json.loads(r["threshold_json"]) if r["threshold_json"] else None)
        alerts = [self._row_to_alert(r) for r in
                  self.conn.execute("SELECT * FROM alerts WHERE run_id=? ORDER BY t_start_s", (run["id"],))]
        return run["id"], AnalysisResult(alerts, scores, json.loads(run["detector_status_json"]),
                                         json.loads(run["timing_json"] or "{}"),
                                         json.loads(run["config_json"] or "{}"))

    @staticmethod
    def _row_to_alert(r: sqlite3.Row) -> Alert:
        return Alert(r["t_start_s"], r["t_end_s"], r["severity"], r["detector"], r["detector_kind"],
                     json.loads(r["signals_json"]), r["explanation"], r["peak_score"], r["threshold"],
                     r["model_version"], json.loads(r["values_json"] or "{}"), r["origin"])

    def list_alerts(self, session_id: int | None = None, severity: str | None = None,
                    detector_kind: str | None = None, latest_run_only: bool = True) -> pd.DataFrame:
        q = ("SELECT a.id, a.session_id, s.source_name, a.run_id, a.t_start_s, a.t_end_s, a.severity, "
             "a.detector, a.detector_kind, a.signals_json, a.explanation, a.peak_score, a.threshold, "
             "a.model_version, a.origin, a.values_json FROM alerts a JOIN sessions s ON s.id = a.session_id WHERE 1=1")
        params: list = []
        if session_id is not None:
            q += " AND a.session_id=?"
            params.append(session_id)
        if severity:
            q += " AND a.severity=?"
            params.append(severity)
        if detector_kind:
            q += " AND a.detector_kind=?"
            params.append(detector_kind)
        if latest_run_only:
            q += " AND a.run_id = (SELECT MAX(id) FROM analysis_runs r WHERE r.session_id = a.session_id)"
        q += " ORDER BY a.session_id DESC, a.t_start_s"
        return pd.read_sql_query(q, self.conn, params=params)

    # ---- settings -----------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        r = self.conn.execute("SELECT value_json FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO settings (key, value_json) VALUES (?,?) "
                              "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json", (key, _j(value)))
