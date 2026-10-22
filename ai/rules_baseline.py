"""Rule-based signal warnings.

These are HEURISTIC, CONFIGURABLE thresholds for a generic passenger car, not
manufacturer specifications and not diagnoses. Rationale per rule is shown
to the user. Rules run on the 1 Hz grid and require a minimum duration so a
single noisy sample does not raise an alert.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ai.events import Alert, segments, snapshot
from core.schema import TIME_COL


@dataclass
class RuleConfig:
    coolant_attention_c: float = 105.0
    coolant_warning_c: float = 110.0
    coolant_min_duration_s: float = 10.0
    running_rpm: float = 500.0                 # engine considered running above this
    charging_low_attention_v: float = 12.6
    charging_low_warning_v: float = 12.0
    voltage_high_warning_v: float = 15.0
    voltage_min_duration_s: float = 10.0
    rpm_high_attention: float = 4500.0
    rpm_min_duration_s: float = 3.0
    stuck_window_s: float = 30.0
    stuck_speed_change_kmh: float = 15.0

    def to_dict(self) -> dict:
        return asdict(self)


RULE_RATIONALE = {
    "coolant_high": "Coolant above the configured limit. Typical warm passenger-car engines run roughly "
                    "85-105 °C (thermostat-controlled); sustained higher values deserve inspection of the "
                    "cooling system by a qualified mechanic. Limit is a configurable heuristic.",
    "charging_voltage_low": "Control-module voltage below the configured limit while the engine runs. With the "
                            "alternator charging, ~13.5-14.8 V is typical; lower values can indicate a charging "
                            "or battery issue OR a logging artefact. Heuristic, not a diagnosis.",
    "voltage_high": "Control-module voltage above the configured limit; may indicate over-charging or a "
                    "measurement artefact. Heuristic.",
    "rpm_high": "Engine speed above a user-configurable limit. Not a fault; limit is vehicle-specific.",
    "rpm_stuck": "RPM reading did not change at all while vehicle speed changed substantially - pattern "
                 "consistent with a frozen reading or logging issue. Not a confirmed sensor failure.",
}


def run_rules(df_1hz: pd.DataFrame, cfg: RuleConfig | None = None, origin: str = "recorded") -> list[Alert]:
    cfg = cfg or RuleConfig()
    t = df_1hz[TIME_COL].to_numpy(dtype=float)
    alerts: list[Alert] = []

    def emit(mask, rule, sev, sigs, min_dur, text_fn, score_col=None, thr=None):
        mask = np.asarray(mask.fillna(False) if isinstance(mask, pd.Series) else mask, dtype=bool)
        for a, b in segments(mask, t, min_duration_s=min_dur, merge_gap_s=2.0):
            seg = df_1hz.iloc[a:b + 1]
            if score_col:
                vals = seg[score_col]
                peak_i = a + int(np.nanargmax(np.abs(vals.to_numpy() - (thr or 0))))
                peak = float(df_1hz[score_col].iloc[peak_i])
            else:
                peak_i, peak = a, None
            alerts.append(Alert(
                t_start_s=float(t[a]), t_end_s=float(t[b]), severity=sev, detector=f"rule:{rule}",
                detector_kind="heuristic_rule", signals=sigs, explanation=text_fn(seg) + " " + RULE_RATIONALE[rule],
                peak_score=peak, threshold=thr, model_version="rules-v1",
                values=snapshot(df_1hz, peak_i, sigs), origin=origin))

    if "coolant_temp_c" in df_1hz:
        c = df_1hz["coolant_temp_c"]
        warn = c >= cfg.coolant_warning_c
        att = (c >= cfg.coolant_attention_c) & ~warn
        emit(warn, "coolant_high", "warning", ["coolant_temp_c"], cfg.coolant_min_duration_s,
             lambda s: f"Coolant reached {s['coolant_temp_c'].max():.1f} °C (≥ {cfg.coolant_warning_c:g}).",
             "coolant_temp_c", cfg.coolant_warning_c)
        emit(att, "coolant_high", "attention", ["coolant_temp_c"], cfg.coolant_min_duration_s,
             lambda s: f"Coolant reached {s['coolant_temp_c'].max():.1f} °C (≥ {cfg.coolant_attention_c:g}).",
             "coolant_temp_c", cfg.coolant_attention_c)

    if "battery_voltage" in df_1hz and "rpm" in df_1hz:
        v, running = df_1hz["battery_voltage"], df_1hz["rpm"] > cfg.running_rpm
        warn = running & (v < cfg.charging_low_warning_v)
        att = running & (v < cfg.charging_low_attention_v) & ~warn
        emit(warn, "charging_voltage_low", "warning", ["battery_voltage", "rpm"], cfg.voltage_min_duration_s,
             lambda s: f"Voltage {s['battery_voltage'].min():.2f} V with engine running.",
             "battery_voltage", cfg.charging_low_warning_v)
        emit(att, "charging_voltage_low", "attention", ["battery_voltage", "rpm"], cfg.voltage_min_duration_s,
             lambda s: f"Voltage {s['battery_voltage'].min():.2f} V with engine running.",
             "battery_voltage", cfg.charging_low_attention_v)
    if "battery_voltage" in df_1hz:
        emit(df_1hz["battery_voltage"] > cfg.voltage_high_warning_v, "voltage_high", "warning",
             ["battery_voltage"], cfg.voltage_min_duration_s,
             lambda s: f"Voltage {s['battery_voltage'].max():.2f} V.", "battery_voltage", cfg.voltage_high_warning_v)

    if "rpm" in df_1hz:
        emit(df_1hz["rpm"] > cfg.rpm_high_attention, "rpm_high", "attention", ["rpm"], cfg.rpm_min_duration_s,
             lambda s: f"RPM up to {s['rpm'].max():.0f}.", "rpm", cfg.rpm_high_attention)

    if "rpm" in df_1hz and "speed_kmh" in df_1hz:
        w = int(cfg.stuck_window_s)
        rpm, spd = df_1hz["rpm"], df_1hz["speed_kmh"]
        flat = rpm.rolling(w).apply(lambda x: float(np.nanmax(x) - np.nanmin(x) == 0), raw=True) == 1
        moving = (spd.rolling(w).max() - spd.rolling(w).min()) >= cfg.stuck_speed_change_kmh
        stuck = flat & moving & (rpm > 0)
        # rolling marks the window END; mark the whole window as affected
        stuck = stuck[::-1].rolling(w, min_periods=1).max()[::-1].astype(bool)
        emit(stuck, "rpm_stuck", "attention", ["rpm", "speed_kmh"], cfg.stuck_window_s,
             lambda s: f"RPM constant at {s['rpm'].iloc[0]:.0f} for {len(s)} s while speed varied.")

    return sorted(alerts, key=lambda a: a.t_start_s)
