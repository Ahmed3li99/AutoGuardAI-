"""Convert the VED weekly CSVs into one 1 Hz file per vehicle.

Usage:  python experiments/prepare_ved.py

Input : data/raw/ved/extracted/VED_*_week.csv (54 files, ~3 GB) + VED_Static_Data_*.xlsx
Output: data/raw/ved/by_vehicle/veh_<id>.pkl  -> {"trips": {trip: 1 Hz DataFrame}, "first_day": {trip: DayNum}}
        data/raw/ved/by_vehicle/index.json   -> per-vehicle type / class / engine / trips / hours

Only columns with a canonical equivalent are kept. 'Absolute Load[%]' (PID 0x43) is NOT mapped to
engine_load_pct, which is calculated load (PID 0x04): the two differ (same choice as the Ford mapping).
Time: t_s = Timestamp(ms) / 1000, relative to the trip start (VED gives no wall-clock time).
"""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd
from common import ROOT

from core.preprocessing.resample import resample_1hz

VED = ROOT / "data" / "raw" / "ved"
OUT = VED / "by_vehicle"
COLS = {"Vehicle Speed[km/h]": "speed_kmh", "Engine RPM[RPM]": "rpm", "MAF[g/sec]": "mass_air_flow_g_per_s",
        "OAT[DegC]": "ambient_air_temp_c", "Short Term Fuel Trim Bank 1[%]": "short_fuel_trim_pct",
        "Long Term Fuel Trim Bank 1[%]": "long_fuel_trim_pct", "Latitude[deg]": "latitude",
        "Longitude[deg]": "longitude"}
KEYS = ["DayNum", "VehId", "Trip", "Timestamp(ms)"]


def static_info() -> dict[int, dict]:
    info = {}
    for name in ("VED_Static_Data_ICE_HEV.xlsx", "VED_Static_Data_PHEV_EV.xlsx"):
        p = VED / name
        if not p.exists():
            continue
        df = pd.read_excel(p)
        for _, r in df.iterrows():
            vtype = r["Vehicle Type"] if "Vehicle Type" in r else r.get("EngineType", "")   # PHEV/EV table
            info[int(r["VehId"])] = {"type": str(vtype).strip(),
                                     "class": str(r.get("Vehicle Class", "")).strip(),
                                     "engine": str(r.get("Engine Configuration & Displacement", "")).strip(),
                                     "transmission": str(r.get("Transmission", "")).strip()}
    return info


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_parts"                  # per-week pieces on disk: the full set does not fit in RAM
    tmp.mkdir(exist_ok=True)
    files = sorted((VED / "extracted").glob("VED_*_week.csv"))
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, usecols=KEYS + list(COLS), dtype={c: "float32" for c in COLS})
        df = df.rename(columns=COLS)
        for vid, g in df.groupby("VehId"):
            g.to_pickle(tmp / f"{int(vid)}__{f.stem}.pkl")
        print(f"[{i}/{len(files)}] {f.name}: {len(df):,} rows", flush=True)
        del df
    parts: dict[int, list] = defaultdict(list)
    for q in tmp.glob("*.pkl"):
        parts[int(q.name.split("__")[0])].append(q)
    meta = static_info()
    index = {}
    for vid, qs in sorted(parts.items()):
        d = pd.concat([pd.read_pickle(q) for q in qs], ignore_index=True)
        trips, first_day = {}, {}
        for trip, g in d.groupby("Trip"):
            g = g.sort_values("Timestamp(ms)")
            g = g.assign(t_s=g["Timestamp(ms)"].to_numpy(float) / 1000.0)
            sig = [c for c in COLS.values() if g[c].notna().any()]
            one, _ = resample_1hz(g, sig)
            trips[int(trip)] = one.astype("float64")      # float32 only to save RAM while reading
            first_day[int(trip)] = float(g["DayNum"].min())
        pd.to_pickle({"trips": trips, "first_day": first_day}, OUT / f"veh_{vid}.pkl")
        has = lambda c: sum(1 for t in trips.values() if c in t and t[c].notna().any())  # noqa: E731
        index[vid] = {**meta.get(vid, {"type": "unknown"}), "trips": len(trips),
                      "hours": round(sum(len(t) for t in trips.values()) / 3600, 2),
                      "trips_with_rpm": has("rpm"), "trips_with_speed": has("speed_kmh")}
        for q in qs:
            q.unlink()
    tmp.rmdir()
    (OUT / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    by_type = defaultdict(lambda: [0, 0.0])
    for v in index.values():
        by_type[v["type"]][0] += 1
        by_type[v["type"]][1] += v["hours"]
    print({k: (n, round(h, 1)) for k, (n, h) in by_type.items()})


if __name__ == "__main__":
    main()
