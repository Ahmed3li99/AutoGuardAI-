"""CSV exports. Every row carries its provenance so a file cannot be mistaken
for a different kind of data once it leaves the app."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai.events import AnalysisResult
from core.schema import NormalizedSession


def export_session_csv(session: NormalizedSession, path: str | Path) -> Path:
    path = Path(path)
    df = session.data.copy()
    df["data_origin"] = session.meta.origin.value
    df["time_basis"] = session.meta.time_basis.value
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def alerts_frame(session: NormalizedSession, result: AnalysisResult, session_db_id: int | None = None) -> pd.DataFrame:
    rows = []
    for a in result.alerts:
        rows.append({
            "session_id": session_db_id, "source": session.meta.source_name,
            "data_origin": a.origin, "time_basis": session.meta.time_basis.value,
            "t_start_s": a.t_start_s, "t_end_s": a.t_end_s, "duration_s": a.t_end_s - a.t_start_s + 1,
            "severity": a.severity, "detector": a.detector, "detector_kind": a.detector_kind,
            "model_version": a.model_version, "peak_score": a.peak_score, "threshold": a.threshold,
            "signals": ";".join(a.signals),
            "values_at_peak": ";".join(f"{k}={v}" for k, v in a.values.items()),
            "explanation": a.explanation})
    return pd.DataFrame(rows, columns=["session_id", "source", "data_origin", "time_basis", "t_start_s",
                                       "t_end_s", "duration_s", "severity", "detector", "detector_kind",
                                       "model_version", "peak_score", "threshold", "signals",
                                       "values_at_peak", "explanation"])


def export_alerts_csv(session: NormalizedSession, result: AnalysisResult, path: str | Path,
                      session_db_id: int | None = None) -> Path:
    path = Path(path)
    alerts_frame(session, result, session_db_id).to_csv(path, index=False, encoding="utf-8")
    return path
