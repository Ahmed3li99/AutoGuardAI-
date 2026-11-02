"""Phase-0 dataset audit: inspects the files actually present under data/raw/
and writes machine-readable evidence to data/processed/audit/ plus
datasets/datasets_manifest.json. Nothing here is estimated - every number is
computed from the files. Missing datasets are reported as NOT OBTAINED.

Usage:  python experiments/audit_datasets.py
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data_sources.files import read_table, sha256_file  # noqa: E402
from core.data_sources.files import RecordedFileSource  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "audit"
TOYOTA_DIR = RAW / "eron93br_carOBD" / "obdiidata"
TOYOTA_MIRROR = RAW / "willkn_odb2" / "obdiidata"
DIESEL_DIR = RAW / "AbouAbdallah-Lounis_OBD-Dataset"
SCANIA_DIR = RAW / "scania_aps"
KIT_DIR = RAW / "kit_obd"
VED_DIR = RAW / "ved"
APM_DIR = RAW / "apmonitor"


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def file_inventory(paths: list[Path]) -> list[dict]:
    return [{"file": str(p.relative_to(ROOT)).replace("\\", "/"), "bytes": p.stat().st_size,
             "sha256": sha256_file(p)} for p in sorted(paths)]


def audit_toyota() -> dict:
    if not TOYOTA_DIR.exists():
        return {"dataset_id": "toyota_carobd", "obtained": False}
    files = sorted(TOYOTA_DIR.glob("*.csv"))
    per_file = []
    header_variants: dict[str, int] = {}
    trailing_comma_files = []
    for f in files:
        first_two = f.read_text(encoding="utf-8", errors="replace").splitlines()[:2]
        n_hdr = len(first_two[0].split(","))
        n_row = len(first_two[1].split(",")) if len(first_two) > 1 else 0
        if n_row == n_hdr + 1:
            trailing_comma_files.append(f.name)
        key = first_two[0]
        header_variants[key] = header_variants.get(key, 0) + 1
        src = RecordedFileSource(f)
        s = src.read_session()
        q = src.quality
        d = s.data
        t = d["t_s"].to_numpy()
        per_file.append({
            "file": f.name, "category": "".join(ch for ch in f.stem if ch.isalpha()),
            "rows": q.rows, "run_time_start_s": float(t[0]), "run_time_end_s": float(t[-1]),
            "duration_s": q.duration_s, "rows_per_s": q.rows_per_second,
            "time_monotonic": q.non_monotonic_time == 0, "n_gaps": q.n_gaps,
            "max_gap_s": q.max_gap_s, "duplicate_rows": q.duplicate_rows,
            "missing_cells": int(d.drop(columns="t_s").isna().sum().sum()),
            "all_missing_signals": [sq.name for sq in q.signals if sq.present == 0],
            "rpm_max": float(d["rpm"].max()), "speed_max": float(d["speed_kmh"].max()),
            "coolant_min": float(d["coolant_temp_c"].min()), "coolant_max": float(d["coolant_temp_c"].max()),
            "voltage_min": float(d["battery_voltage"].min()), "voltage_median": float(d["battery_voltage"].median()),
            "frac_rows_rpm_zero": float((d["rpm"] == 0).mean()),
            "trailing_delimiter": f.name in trailing_comma_files,
        })
    pf = pd.DataFrame(per_file)
    all_data = pd.concat([RecordedFileSource(f).read_session().data.assign(file=f.name) for f in files])
    sig_cols = [c for c in all_data.columns if c not in ("t_s", "file")]
    stats = all_data[sig_cols].describe(percentiles=[.01, .5, .99]).T.round(3)
    mirror_identical = None
    if TOYOTA_MIRROR.exists():
        a = {p.name: sha256_file(p) for p in TOYOTA_MIRROR.glob("*.csv")}
        b = {p.name: sha256_file(p) for p in files}
        mirror_identical = a == b
    by_cat = pf.groupby("category").agg(trips=("file", "count"), rows=("rows", "sum"),
                                        hours=("duration_s", lambda x: round(x.sum() / 3600, 2)))
    return {
        "dataset_id": "toyota_carobd", "obtained": True,
        "source_url": "https://github.com/eron93br/carOBD",
        "git_commit": git_commit(TOYOTA_DIR.parent),
        "mirror": {"url": "https://github.com/willkn/odb2", "git_commit": git_commit(TOYOTA_MIRROR.parent),
                   "obdiidata_byte_identical": mirror_identical},
        "n_files": len(files), "total_rows": int(pf.rows.sum()),
        "total_hours": round(float(pf.duration_s.sum()) / 3600, 2),
        "by_category": by_cat.reset_index().to_dict(orient="records"),
        "header_variants": [{"n_files": v, "n_columns": len(k.split(",")), "header": k}
                            for k, v in header_variants.items()],
        "files_with_trailing_delimiter": len(trailing_comma_files),
        "files_without_trailing_delimiter": sorted(set(f.name for f in files) - set(trailing_comma_files)),
        "all_time_monotonic": bool(pf.time_monotonic.all()),
        "rows_per_s_range": [float(pf.rows_per_s.min()), float(pf.rows_per_s.max())],
        "signal_stats": stats.reset_index().rename(columns={"index": "signal"}).to_dict(orient="records"),
        "files_with_all_missing_signals": pf[pf.all_missing_signals.str.len() > 0][["file", "all_missing_signals"]].to_dict(orient="records"),
        "per_file": per_file,
        "inventory": file_inventory(files),
    }


def audit_diesel() -> dict:
    if not DIESEL_DIR.exists():
        return {"dataset_id": "dieselobd", "obtained": False}
    out = {"dataset_id": "dieselobd", "obtained": True,
           "source_url": "https://github.com/AbouAbdallah-Lounis/OBD-Dataset",
           "git_commit": git_commit(DIESEL_DIR),
           "inventory": file_inventory([p for p in DIESEL_DIR.iterdir() if p.is_file()]),
           "splits": {}}
    dtc_cols = None
    sigs = {}
    for split, fname in (("train", "MASTER_TRAIN_X.xlsx"), ("test", "MASTER_TEST_X.xlsx")):
        df = read_table(DIESEL_DIR / fname)
        dtc_cols = [c for c in df.columns if c[:1] in ("P", "U", "B", "C") and c[1:].isdigit()]
        pid_cols = [c for c in df.columns if c not in dtc_cols and c != "Mode"]
        sig = df[dtc_cols].astype(int).astype(str).agg("".join, axis=1)
        sig_names = {s: "+".join(c for c, b in zip(dtc_cols, s) if b == "1") for s in sig.unique()}
        sigs[split] = {sig_names[k]: int(v) for k, v in sig.value_counts().items()}
        out["splits"][split] = {
            "file": fname, "rows": len(df), "columns": list(df.columns),
            "pid_columns": pid_cols, "dtc_columns": dtc_cols,
            "has_vehicle_id_column": "vehicle_id" in df.columns,
            "has_time_column": any(c.lower() in ("time", "timestamp", "t") for c in df.columns),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "mode_counts": {str(k): int(v) for k, v in df["Mode"].value_counts().sort_index().items()},
            "pid_stats": df[pid_cols].describe(percentiles=[.01, .5, .99]).T.round(3).reset_index()
                           .rename(columns={"index": "pid"}).to_dict(orient="records"),
            "dtc_signature_counts": sigs[split],
        }
    out["dtc_signatures_shared_by_train_and_test"] = sorted(set(sigs["train"]) & set(sigs["test"]))
    out["n_distinct_dtc_signatures"] = len(set(sigs["train"]) | set(sigs["test"]))
    return out


def audit_scania() -> dict:
    tr, te = SCANIA_DIR / "aps_failure_training_set.csv", SCANIA_DIR / "aps_failure_test_set.csv"
    if not tr.exists():
        return {"dataset_id": "scania_aps", "obtained": False}
    res = {"dataset_id": "scania_aps", "obtained": True,
           "source_url": "https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks",
           "inventory": file_inventory([p for p in SCANIA_DIR.iterdir() if p.is_file()])}
    for split, p in (("train", tr), ("test", te)):
        lines = p.read_text(encoding="utf-8").splitlines()
        hdr = next(i for i, ln in enumerate(lines) if ln.startswith("class,"))
        df = pd.read_csv(p, skiprows=hdr, na_values=["na"])
        res[split] = {"preamble_lines": hdr, "rows": len(df), "n_features": df.shape[1] - 1,
                      "class_counts": df["class"].value_counts().to_dict(),
                      "missing_rate": round(float(df.drop(columns="class").isna().mean().mean()), 4),
                      "feature_name_examples": list(df.columns[1:8])}
    return res


def audit_kit() -> dict:
    base = {"dataset_id": "kit_obd", "source_url": "https://radar.kit.edu/radar/en/dataset/bCtGxdTklQlfQcAq",
            "doi": "10.35097/1130", "license": "CC BY 4.0 (per landing page)",
            "access": "Download requires accepting RADAR4KIT terms of use in the browser."}
    csvs = sorted(KIT_DIR.rglob("*.csv")) if KIT_DIR.exists() else []
    if not csvs:
        return {**base, "obtained": False}
    files = []
    for p in csvs:
        try:
            df = read_table(p)
            files.append({"file": p.name, "rows": len(df), "columns": list(df.columns)})
        except Exception as e:
            files.append({"file": p.name, "error": str(e)})
    return {**base, "obtained": True, "files": files, "inventory": file_inventory(csvs)}


def audit_ved() -> dict:
    """VED: inventory of the archives + static tables; per-vehicle numbers come from the
    1 Hz per-vehicle files written by experiments/prepare_ved.py (run that first)."""
    if not (VED_DIR / "VED_Static_Data_ICE_HEV.xlsx").exists():
        return {"dataset_id": "ved", "obtained": False}
    res = {"dataset_id": "ved", "obtained": True, "source_url": "https://github.com/gsoh/VED",
           "inventory": file_inventory([p for p in VED_DIR.iterdir() if p.is_file()])}
    weeks = sorted((VED_DIR / "extracted").glob("VED_*_week.csv"))
    res["weekly_csv_files"] = len(weeks)
    res["weekly_csv_bytes"] = sum(p.stat().st_size for p in weeks)
    if weeks:
        res["columns"] = list(pd.read_csv(weeks[0], nrows=0).columns)
    static = pd.concat([pd.read_excel(p) for p in VED_DIR.glob("VED_Static_Data_*.xlsx")], ignore_index=True)
    res["vehicles_in_static_tables"] = int(len(static))
    vt = static["Vehicle Type"].fillna(static.get("EngineType")) if "EngineType" in static else static["Vehicle Type"]
    res["vehicle_types"] = vt.astype(str).str.strip().value_counts().to_dict()
    idx = VED_DIR / "by_vehicle" / "index.json"
    if idx.exists():
        ix = json.loads(idx.read_text(encoding="utf-8"))
        res["vehicles_with_dynamic_data"] = len(ix)
        res["trips"] = sum(v["trips"] for v in ix.values())
        res["hours_1hz"] = round(sum(v["hours"] for v in ix.values()), 1)
        by = {}
        for v in ix.values():
            b = by.setdefault(v.get("type", "unknown"), {"vehicles": 0, "trips": 0, "hours": 0.0})
            b["vehicles"] += 1
            b["trips"] += v["trips"]
            b["hours"] = round(b["hours"] + v["hours"], 2)
        res["by_type"] = by
        res["vehicles_with_rpm"] = sum(1 for v in ix.values() if v["trips_with_rpm"])
    res["no_coolant_voltage_throttle_map"] = True
    res["time_column"] = "Timestamp(ms) relative to trip start; DayNum = days since 2017-11-01 (fractional)"
    return res


def audit_apmonitor() -> dict:
    files = {"auto_warmup.txt": "ford", "automotive.txt": "ford", "auto_trip/auto_trip.csv": "ford",
             "auto_iowa.txt": "chrysler"}
    if not (APM_DIR / "auto_iowa.txt").exists():
        return {"dataset_id": "apmonitor_automotive", "obtained": False}
    res = {"dataset_id": "apmonitor_automotive", "obtained": True,
           "source_url": "https://apmonitor.com/pds/index.php/Main/AutomotiveMonitoring",
           "inventory": file_inventory([APM_DIR / f for f in files] + [APM_DIR / "auto_trip.zip"]), "files": {}}
    for f, car in files.items():
        s = RecordedFileSource(APM_DIR / f).read_session()
        d = s.data
        cover = {c: round(float(d[c].notna().mean()), 3) for c in s.signals}
        res["files"][f] = {"car": car, "mapping_id": s.meta.mapping_id, "rows": int(len(d)),
                           "duration_s": round(s.duration_s, 1), "signals": s.signals, "row_coverage": cover,
                           "max": {c: round(float(d[c].max()), 1) for c in s.signals}}
    return res


STATIC = {
    "toyota_carobd": {
        "name": "carOBD (Toyota Etios 2014)", "provider": "GitHub user eron93br (master-thesis dataset)",
        "citation": "'Detecting Anomalies in the Engine Coolant Sensor Using One-Class Classifiers', IEEE, "
                    "https://ieeexplore.ieee.org/abstract/document/8891367 (cited by the repo README; author "
                    "names to be copied from the IEEE record - not verified offline).",
        "license": "No LICENSE file. README: 'open-source repository, free to everyone' and asks for citation. "
                   "Redistribution terms are informal - treat bundled sample as 'cite, confirm before public release'.",
        "ground_truth": "none (no fault or anomaly labels; file name encodes operating mode only)",
        "decision": "ACCEPT", "uses": ["recorded replay", "unlabelled anomaly detection (train/calibrate/test by trip)",
                                       "synthetic-injection stress tests on held-out trips", "bundled demo sample"],
    },
    "dieselobd": {
        "name": "DieselOBD", "provider": "Karim Lounis, Hadjer Mekki, Wissam Hadjab",
        "citation": "K. Lounis, H. Mekki, W. Hadjab. DieselOBD: A Real-World Dataset of OBD-II Parameters and "
                    "Diagnostic Trouble Codes for Diesel Vehicle Fault Diagnosis. 2025.",
        "license": "CC BY 4.0 (stated in README; no separate LICENSE file)",
        "ground_truth": "vehicle-level DTC tags only (constant per vehicle); not per-sample diagnoses",
        "decision": "RESTRICTED USE",
        "uses": ["import/mapping test of unit-inferred XLSX", "replay with synthetic playback clock",
                 "optional leave-one-signature-out experiment only - no fault-classifier enabled in app"],
    },
    "ved": {
        "name": "Vehicle Energy Dataset (VED)", "provider": "G. Oh, D. J. LeBlanc, H. Peng (University of Michigan)",
        "citation": "G. Oh, D. J. LeBlanc, H. Peng. Vehicle Energy Dataset (VED), A Large-scale Dataset for Vehicle "
                    "Energy Consumption Research. IEEE T-ITS, 2020. doi:10.1109/TITS.2020.3035596",
        "license": "Apache License 2.0 (LICENSE file in the repository)",
        "ground_truth": "none (no fault labels); vehicle type / class / engine from the static tables",
        "decision": "ACCEPT (cross-car validation only)",
        "uses": ["cross-car validation of the drivetrain model and 'learn your car' (RPM + speed only)",
                 "consistency-check behaviour on ICE / HEV / PHEV / EV"],
    },
    "apmonitor_automotive": {
        "name": "APMonitor Automotive Monitoring (2014 Ford Focus, 2021 Chrysler Pacifica)",
        "provider": "APMonitor / BYU PRISM course material (J. Hedengren)",
        "citation": "APMonitor, Automotive Monitoring, https://apmonitor.com/pds/index.php/Main/AutomotiveMonitoring",
        "license": "No license stated on the page: used locally for evaluation only, NOT bundled or redistributed.",
        "ground_truth": "none; the long Ford file contains real glitch values (e.g. RPM 11,596, coolant 190 C)",
        "decision": "RESTRICTED USE (local evaluation only)",
        "uses": ["cross-car validation (full + drivetrain models)", "real sensor glitches vs the consistency check",
                 "import tests for Car Scanner / OBD Fusion style logs"],
    },
    "scania_aps": {
        "name": "APS Failure at Scania Trucks", "provider": "Scania CV AB via UCI ML Repository",
        "citation": "APS Failure at Scania Trucks. Scania CV AB, donated by T. Lindgren and J. Biteus, 2016. "
                    "UCI Machine Learning Repository, dataset 421.",
        "license": "aps_failure_description.txt states GNU GPL v3 (Copyright 2016 Scania CV AB). "
                   "UCI page license not re-verified.",
        "ground_truth": "binary per-record APS failure label (anonymized truck features)",
        "decision": "RESTRICTED USE",
        "uses": ["standalone truck-APS classification benchmark only - never mixed with car OBD data"],
    },
    "kit_obd": {
        "name": "Automotive OBD-II Dataset (KIT)", "provider": "Karlsruhe Institute of Technology (RADAR4KIT)",
        "citation": "Automotive OBD-II Dataset (2023). KIT. https://doi.org/10.35097/1130",
        "license": "CC BY 4.0 (landing page)",
        "ground_truth": "road-condition labels in file names (normal/free/busy) - NOT mechanical faults",
        "decision": "PENDING (not obtained)",
        "uses": ["candidate second replay/benchmark source once manually downloaded"],
    },
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    results = {"toyota_carobd": audit_toyota(), "dieselobd": audit_diesel(),
               "scania_aps": audit_scania(), "kit_obd": audit_kit(), "ved": audit_ved(),
               "apmonitor_automotive": audit_apmonitor()}
    for k, v in results.items():
        (OUT / f"{k}.json").write_text(json.dumps(v, indent=2, default=str), encoding="utf-8")
    manifest = {"generated": today, "generator": "experiments/audit_datasets.py", "datasets": []}
    for k, v in results.items():
        entry = {"dataset_id": k, **STATIC[k], "source_url": v.get("source_url"),
                 "obtained": v.get("obtained", False), "audit_file": f"data/processed/audit/{k}.json"}
        if v.get("git_commit"):
            entry["git_commit"] = v["git_commit"]
        if v.get("doi"):
            entry["doi"] = v["doi"]
        if v.get("inventory"):
            entry["n_files"] = len(v["inventory"])
            entry["total_bytes"] = sum(i["bytes"] for i in v["inventory"])
        if k == "kit_obd" and v.get("obtained"):
            entry["decision"] = "OBTAINED - re-run audit review"
        manifest["datasets"].append(entry)
    (ROOT / "datasets" / "datasets_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    t = results["toyota_carobd"]
    if t.get("obtained"):
        print(f"toyota: {t['n_files']} files, {t['total_rows']} rows, {t['total_hours']} h, "
              f"monotonic={t['all_time_monotonic']}, rows/s={t['rows_per_s_range']}")
    d = results["dieselobd"]
    if d.get("obtained"):
        print(f"diesel: train={d['splits']['train']['rows']} test={d['splits']['test']['rows']} "
              f"signatures={d['n_distinct_dtc_signatures']} shared={len(d['dtc_signatures_shared_by_train_and_test'])}")
    s = results["scania_aps"]
    if s.get("obtained"):
        print(f"scania: train={s['train']['class_counts']} test={s['test']['class_counts']}")
    print(f"kit: obtained={results['kit_obd']['obtained']}")
    v = results["ved"]
    if v.get("obtained"):
        print(f"ved: {v.get('vehicles_with_dynamic_data')} vehicles, {v.get('trips')} trips, {v.get('hours_1hz')} h, "
              f"types={v.get('by_type')}")
    a = results["apmonitor_automotive"]
    if a.get("obtained"):
        for f, x in a["files"].items():
            print(f"apmonitor {f}: {x['car']} {x['duration_s']} s, {len(x['signals'])} signals")
    print(f"wrote {OUT} and datasets/datasets_manifest.json")


if __name__ == "__main__":
    main()
