"""Run every available detector on a session and collect evidence-backed alerts."""
from __future__ import annotations

import logging
import time

import numpy as np

from ai.events import AnalysisResult, DetectorScores
from ai.model_registry import ModelRegistry
from ai.rules_baseline import RuleConfig, run_rules
from ai.thresholds import scores_to_alerts
from core.preprocessing.resample import resample_1hz
from core.schema import TIME_COL, NormalizedSession, TimeBasis

log = logging.getLogger(__name__)
MIN_SECONDS_FOR_MODELS = 60


def run_analysis(session: NormalizedSession, registry: ModelRegistry,
                 rule_cfg: RuleConfig | None = None) -> AnalysisResult:
    rule_cfg = rule_cfg or RuleConfig()
    origin = session.meta.origin.value
    df1, rinfo = resample_1hz(session.data, session.signals)
    status: dict[str, dict] = {"rules": {"status": "ok", "version": "rules-v1", "kind": "heuristic_rule"}}
    timing: dict[str, float] = {}
    scores: dict[str, DetectorScores] = {}
    alerts = []
    notes = []
    if session.meta.time_basis == TimeBasis.SYNTHETIC_CLOCK and session.meta.origin.value == "recorded":
        notes.append("Source has no recorded time; durations below refer to the synthetic playback clock.")

    t0 = time.perf_counter()
    alerts += run_rules(df1, rule_cfg, origin)
    timing["rules"] = time.perf_counter() - t0
    t_list = df1[TIME_COL].tolist()

    for name, getter in (("isolation_forest", registry.iforest), ("moment", registry.moment)):
        model = getter()
        st = dict(registry.status.get(name, {"status": "unavailable", "reason": "not loaded"}))
        if model is None:
            status[name] = st
            continue
        miss = model.missing_signals(list(df1.columns))
        if miss:
            status[name] = {**st, "status": "unavailable", "reason": f"Required signals missing: {miss}"}
            continue
        if model.threshold is None:
            status[name] = {**st, "status": "unavailable", "reason": "No calibrated threshold."}
            continue
        n_ok = int(df1[model.signals].notna().all(axis=1).sum())
        if n_ok < MIN_SECONDS_FOR_MODELS:
            status[name] = {**st, "status": "insufficient_data",
                            "reason": f"Only {n_ok} s with all required signals (< {MIN_SECONDS_FOR_MODELS})."}
            continue
        note = ""
        trained = model.meta.get("calibrated_on_dataset")
        if trained and session.meta.dataset_id != trained:
            note = (f"Threshold was calibrated on '{trained}' data; this session is "
                    f"'{session.meta.dataset_id or 'unknown'}', so the score scale is UNCALIBRATED here.")
            st["calibration"] = "uncalibrated_for_this_data"
        else:
            st["calibration"] = "calibrated"
        try:
            t1 = time.perf_counter()
            out = model.score(df1)
            sc, contrib = out[0], out[1]
            timing[name] = time.perf_counter() - t1
            if len(out) > 2:
                st["device"] = out[2].get("device")
        except Exception as e:
            log.exception("%s scoring failed", name)
            status[name] = {**st, "status": "error", "reason": f"{type(e).__name__}: {e}"}
            continue
        scores[name] = DetectorScores(t_list, [None if not np.isfinite(v) else float(v) for v in sc],
                                      model.threshold.to_dict())
        alerts += scores_to_alerts(df1, sc, model.threshold, f"model:{name}", model.version,
                                   contrib, origin, note=note)
        status[name] = {**st, "status": "ok", "note": note} if note else {**st, "status": "ok"}

    alerts.sort(key=lambda a: (a.t_start_s, a.detector))
    return AnalysisResult(alerts, scores, status, timing,
                          {"rules": rule_cfg.to_dict(), "resample": {"seconds": rinfo.seconds,
                                                                     "filled_cells": rinfo.filled_cells},
                           "notes": notes})


def overall_status(result: AnalysisResult | None, session: NormalizedSession) -> str:
    """Normal / Attention / Warning / Insufficient Data / Model Unavailable.

    'Warning' is reserved for interpretable rule criteria; statistical anomalies
    alone (however high their score) raise at most 'Attention'."""
    if session.duration_s < MIN_SECONDS_FOR_MODELS or not session.signals:
        return "Insufficient Data"
    if result is None:
        return "Not Analysed"
    if any(a.severity == "warning" and a.detector_kind == "heuristic_rule" for a in result.alerts):
        return "Warning"
    if result.alerts:
        return "Attention"
    if not any(s.get("status") == "ok" for k, s in result.detector_status.items() if k != "rules"):
        return "Normal (rules only - models unavailable)"
    return "Normal"
