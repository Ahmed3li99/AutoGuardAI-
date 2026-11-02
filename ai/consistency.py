"""Sensor-consistency checks: "is the car sick, or is the sensor lying?"

These checks do not judge the car; they judge whether the DATA tell a
physically consistent story. Each check flags a pattern that a real car can
hardly produce, so it more likely points to a measurement or logging problem.

Thresholds were set well above the largest 1-second changes seen in all 129
real carOBD trips (150,118 s at 1 Hz, see docs/MODEL_CARD.md):
    coolant  max |change| 3 °C/s   -> implausible above 8 °C/s
    intake air max 3 °C/s          -> implausible above 10 °C/s
    speed    max 33 km/h/s         -> implausible above 45 km/h/s (> 1.25 g)
    speed > 20 km/h with RPM = 0 never occurred (0 of 150,118 s)

Implausible values / spikes (RPM, MAF, barometric pressure): limit = 1.25 x the largest real value and
the largest real 1-second change in the DERIVATION data only - carOBD train + calibration trips and the
VED dev petrol cars (experiments/derive_plausibility_limits.py). Measured afterwards on held-out data
(Toyota test, VED test cars and hybrids, Ford, Chrysler): docs/EVALUATION.md §G4.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ai.events import Alert, segments, snapshot
from core.schema import TIME_COL

KIND = "data_consistency"


# signal: (value limit, 1-s change limit, largest real value, largest real 1-s change)
SPIKE_LIMITS = {
    "rpm": (8256.0, 5710.0, 6605.0, 4568.0),                     # VED dev cars (Toyota max 3,151)
    "mass_air_flow_g_per_s": (324.1, 295.6, 259.3, 236.5),      # VED dev cars (no MAF in carOBD)
    "baro_pressure_kpa": (128.75, 3.75, 103.0, 3.0),            # carOBD (no baro in VED)
}


@dataclass
class ConsistencyConfig:
    coolant_jump_c_per_s: float = 8.0
    iat_jump_c_per_s: float = 10.0
    speed_jump_kmh_per_s: float = 45.0
    moving_speed_kmh: float = 20.0
    moving_rpm0_min_s: float = 3.0
    dropout_min_s: float = 10.0          # one signal missing while the others keep reporting
    # hybrids / plug-ins switch the engine off while driving: 0 RPM at speed is normal there
    # (VED: 56 such alerts/h on 93 real hybrids vs 0 on 264 petrol cars)
    engine_can_stop_while_moving: bool = False
    check_spikes: bool = True
    check_dropouts: bool = True
    # frozen reading: the exact same value while moving (> 20 km/h) although the other signal changed.
    # Shortest limits with <= 0.05 alerts/h on 1,420 h of derivation data (Toyota train+calibration,
    # VED dev cars): RPM 8 s (0.012/h), speed 20 s (0.039/h). experiments/derive_stuck_limits.py
    rpm_stuck_s: float = 8.0
    # engine running at any speed (also parked/slow): non-zero RPM exactly constant >= 20 s happened
    # 0.048 times/h on the same 1,420 h of derivation data (shortest limit within the 0.05/h budget)
    rpm_stuck_any_speed_s: float = 20.0
    speed_stuck_s: float = 20.0
    check_stuck: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


TEXT = {
    "jump": "{sig} jumped by {d:.1f} {u} within one second ({a:.1f} → {b:.1f}). A real {what} cannot change "
            "that fast; this points to a measurement or logging problem.",
    "moving_rpm0": "Speed was {v:.0f} km/h while RPM read 0 for {n:.0f} s. A moving car with a stopped engine "
                   "never appeared in the real trips; the RPM or speed reading is probably wrong (unless the car is a hybrid).",
    "dropout": "{sig} stopped reporting for {n:.0f} s while the other sensors kept reporting.",
    "stuck_any": "rpm stayed at exactly {v:g} for {n:.0f} s with the engine running. A working RPM sensor "
                 "always moves a little; the reading is probably frozen.",
    "stuck": "{sig} stayed at exactly {v:g} for {n:.0f} s while the car was moving and {other} changed by "
             "{d:.0f}. A working sensor never stays that still; the reading is probably frozen.",
    "spike": "{sig} read {v:.1f} {u} (1-second change {d:.1f} {u}). The largest real value in the reference "
             "driving data was {mx:g} {u} and the largest 1-second change {mxd:g} {u}; this is most likely a "
             "reading or logging glitch, not the engine.",
}
SPIKE_UNITS = {"rpm": "rpm", "mass_air_flow_g_per_s": "g/s", "baro_pressure_kpa": "kPa"}
WHAT = {"coolant_temp_c": "engine coolant", "intake_air_temp_c": "intake air", "speed_kmh": "car"}
UNITS = {"coolant_temp_c": "°C", "intake_air_temp_c": "°C", "speed_kmh": "km/h"}


def run_consistency(df: pd.DataFrame, cfg: ConsistencyConfig | None = None, origin: str = "recorded") -> list[Alert]:
    cfg = cfg or ConsistencyConfig()
    t = df[TIME_COL].to_numpy(dtype=float)
    out: list[Alert] = []
    # 1) physically impossible 1-second jumps
    for sig, lim in (("coolant_temp_c", cfg.coolant_jump_c_per_s), ("intake_air_temp_c", cfg.iat_jump_c_per_s),
                     ("speed_kmh", cfg.speed_jump_kmh_per_s)):
        if sig not in df:
            continue
        v = df[sig].to_numpy(dtype=float)
        dv = np.abs(np.diff(v, prepend=np.nan))
        dt = np.diff(t, prepend=np.nan)
        rate = np.where(dt > 0, dv / dt, np.nan)
        for a, b in segments(np.nan_to_num(rate) > lim, t, 0, 5):
            i = a + int(np.nanargmax(rate[a:b + 1]))
            others = [s for s in ("rpm", "speed_kmh", "engine_load_pct") if s in df and s != sig]
            out.append(Alert(float(t[max(i - 1, 0)]), float(t[b]), "attention", f"consistency:{sig}_jump", KIND,
                             [sig] + others[:1],
                             TEXT["jump"].format(sig=sig, d=dv[i], u=UNITS[sig], a=v[i - 1], b=v[i], what=WHAT[sig]),
                             float(rate[i]), lim, "consistency-v1", snapshot(df, i, [sig] + others[:1]), origin))
    # 2) moving fast with the engine at 0 RPM
    if "speed_kmh" in df and "rpm" in df and not cfg.engine_can_stop_while_moving:
        m = (df["speed_kmh"] > cfg.moving_speed_kmh) & (df["rpm"] == 0)
        for a, b in segments(m.to_numpy(), t, cfg.moving_rpm0_min_s, 2):
            out.append(Alert(float(t[a]), float(t[b]), "attention", "consistency:moving_rpm0", KIND,
                             ["rpm", "speed_kmh"],
                             TEXT["moving_rpm0"].format(v=float(df["speed_kmh"].iloc[a:b + 1].mean()), n=t[b] - t[a] + 1),
                             None, None, "consistency-v1", snapshot(df, a, ["rpm", "speed_kmh"]), origin))
    # 3) implausible values / spikes (glitches inside the encodable PID range)
    for sig, (vlim, dlim, mx, mxd) in (SPIKE_LIMITS.items() if cfg.check_spikes else ()):
        if sig not in df:
            continue
        v = df[sig].to_numpy(dtype=float)
        dv = np.abs(np.diff(v, prepend=np.nan))
        dt = np.diff(t, prepend=np.nan)
        rate = np.where(dt > 0, dv / dt, np.nan)
        bad = (np.nan_to_num(v, nan=0.0) > vlim) | (np.nan_to_num(rate, nan=0.0) > dlim)
        for a, b in segments(bad, t, 0, 5):
            i = a + int(np.nanargmax(np.nan_to_num(v[a:b + 1], nan=-np.inf)))
            vals = snapshot(df, i, [sig])
            vals[f"{sig}__step"] = round(float(np.nanmax(dv[a:b + 1])), 2)
            out.append(Alert(float(t[max(a - 1, 0)]), float(t[b]), "attention", f"consistency:{sig}_spike", KIND,
                             [sig], TEXT["spike"].format(sig=sig, v=v[i], d=float(np.nanmax(dv[a:b + 1])),
                                                         u=SPIKE_UNITS[sig], mx=mx, mxd=mxd),
                             float(v[i]), vlim, "consistency-v2", vals, origin))
    # 4) frozen reading while driving
    stuck = (("rpm", "speed_kmh", 5.0, cfg.rpm_stuck_s), ("speed_kmh", "rpm", 200.0, cfg.speed_stuck_s))
    for sig, other, need, lim in (stuck if cfg.check_stuck else ()):
        if sig not in df or other not in df or "speed_kmh" not in df:
            continue
        v, o = df[sig].to_numpy(float), df[other].to_numpy(float)
        spd = df["speed_kmh"].to_numpy(float)
        start = 0
        for i in range(1, len(v) + 1):
            if i < len(v) and np.isfinite(v[i]) and v[i] == v[i - 1]:
                continue
            a, b = start, i - 1
            start = i
            if b - a + 1 < lim or not np.isfinite(v[a]) or (sig == "rpm" and v[a] == 0):
                continue          # RPM 0 while moving = engine off: check 2 (and the hybrid exemption)
            seg_o = o[a:b + 1]
            if (np.nanmin(spd[a:b + 1]) <= cfg.moving_speed_kmh or not np.isfinite(seg_o).any()
                    or np.nanmax(seg_o) - np.nanmin(seg_o) < need):
                continue
            d = float(np.nanmax(seg_o) - np.nanmin(seg_o))
            out.append(Alert(float(t[a]), float(t[b]), "attention", f"consistency:{sig}_stuck", KIND, [sig],
                             TEXT["stuck"].format(sig=sig, v=v[a], n=t[b] - t[a] + 1, other=other, d=d),
                             float(b - a + 1), lim, "consistency-v3", snapshot(df, a, [sig, other]), origin))
    # 4b) frozen non-zero RPM at any speed (parked / slow driving), longer limit
    if cfg.check_stuck and "rpm" in df:
        v = df["rpm"].to_numpy(float)
        flagged = [(x.t_start_s, x.t_end_s) for x in out if x.detector == "consistency:rpm_stuck"]
        start = 0
        for i in range(1, len(v) + 1):
            if i < len(v) and np.isfinite(v[i]) and v[i] == v[i - 1]:
                continue
            a, b = start, i - 1
            start = i
            if b - a + 1 < cfg.rpm_stuck_any_speed_s or not np.isfinite(v[a]) or v[a] == 0:
                continue
            if any(s0 <= t[b] and e0 >= t[a] for s0, e0 in flagged):
                continue                      # already reported by the driving rule
            out.append(Alert(float(t[a]), float(t[b]), "attention", "consistency:rpm_stuck", KIND, ["rpm"],
                             TEXT["stuck_any"].format(v=v[a], n=t[b] - t[a] + 1),
                             float(b - a + 1), cfg.rpm_stuck_any_speed_s, "consistency-v3",
                             snapshot(df, a, ["rpm"]), origin))
    # 5) one signal drops out while the others keep reporting
    sigs = [c for c in df.columns if c != TIME_COL and df[c].notna().any()] if cfg.check_dropouts else []
    for sig in sigs:
        others = [c for c in sigs if c != sig]
        if not others:
            continue
        # Do not interpret a signal that has not started reporting yet as a dropout.
        seen = df[sig].notna().cummax()
        m = seen & df[sig].isna() & df[others].notna().any(axis=1)
        for a, b in segments(m.to_numpy(), t, cfg.dropout_min_s, 2):
            out.append(Alert(float(t[a]), float(t[b]), "attention", f"consistency:{sig}_dropout", KIND, [sig],
                             TEXT["dropout"].format(sig=sig, n=t[b] - t[a] + 1), None, None, "consistency-v1",
                             {}, origin))
    return sorted(out, key=lambda x: x.t_start_s)
