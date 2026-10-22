# Dataset Audit (Phase 0)

Generated from the files actually downloaded on **2026-10-22** by `experiments/audit_datasets.py`.
Machine-readable evidence: `data/processed/audit/*.json`, `datasets/datasets_manifest.json`
(includes per-file SHA-256). Numbers below are copied from those files.

## 1. Comparison

| | carOBD (Toyota) | DieselOBD | Scania APS | KIT OBD-II |
|---|---|---|---|---|
| Obtained | ✅ git clone `eron93br/carOBD` @ `bf09e1c` | ✅ git clone @ `a9c96dc` | ✅ UCI zip (56.6 MB) | ❌ not obtained (see §5) |
| Files used | 129 CSV trips | MASTER_TRAIN_X.xlsx, MASTER_TEST_X.xlsx | train/test CSV | – |
| Rows | 304,299 | 118,477 + 33,838 | 60,000 + 16,000 | – |
| Vehicles | 1 (Toyota Etios 2014, 1496 cc) | 8 diesel vehicles per README; **no vehicle_id column** | anonymized trucks | – |
| Time column | engine run time (PID 0x1F), relative, 1 s resolution | **none** | none (not a time series) | – |
| Labels | none (file name = operating mode) | vehicle-level DTC tags | binary APS failure per record | road condition (not faults) |
| License | no LICENSE file; README "free to everyone", cite paper | CC BY 4.0 (README) | GPL v3 (description file) | CC BY 4.0 (landing page) |
| **Decision** | **ACCEPT** | **RESTRICTED USE** | **RESTRICTED USE** | **PENDING** |

## 2. carOBD (Toyota Etios 2014)

**Provenance.** https://github.com/eron93br/carOBD, commit `bf09e1c00f8443212fdbf4b09e8b2a468badd775`.
https://github.com/willkn/odb2 (commit `ee61518`) is a fork. Its `obdiidata/` is **byte-identical**
(SHA-256 compared), so it is a mirror, not an independent source. Logger: Carloop (Particle
Electron) on the OBD-II port. Citation requested: IEEE paper
https://ieeexplore.ieee.org/abstract/document/8891367 ("Detecting Anomalies in the Engine Coolant
Sensor Using One-Class Classifiers"). Copy the author list from the IEEE record. It was not
verified offline.

**Contents.**

| category (file prefix) | trips | rows | hours | meaning per README |
|---|---|---|---|---|
| idle | 47 | 72,226 | 9.83 | parked, engine on |
| live | 39 | 97,281 | 13.45 | work → home route |
| long | 12 | 60,947 | 8.32 | long trips |
| drive | 13 | 39,779 | 5.43 | high-speed roads |
| ufpe | 18 | 34,066 | 4.64 | university campus, low speed |
| **total** | **129** | **304,299** | **41.66** | |

27 columns, identical header in all files (`ENGINE_RUN_TINE ()`, with the source's typo, through
`WARM_UPS_SINCE_CODES_CLEARED ()`).

**Findings that change how the data must be read:**
1. **Trailing delimiter.** 121 of 129 files end every data row with `,` (28 fields vs 27 header
   names). A default `pandas.read_csv` silently shifts *every* column by one: RPM appears as run
   time, coolant as load, and so on. The 8 files `live5`–`live12` have no trailing comma. The reader
   uses `index_col=False`, and a regression test (`test_toyota_trailing_delimiter_alignment`)
   guards this.
2. **Sampling.** The README says 1 Hz. The files contain **1.83–2.07 rows per second** of run time.
   The run-time clock has 1 s resolution, so rows within a second carry no timing information.
   Models therefore use a 1 Hz grid (last row per second). Replay uses the recorded clock.
3. **Time.** Run time is monotonic in all 129 files and is a *relative* clock (seconds since engine
   start). No calendar date exists, so none is shown or fabricated.
4. **Duplicates.** Fully duplicated rows make up 15.7–31.6 % per file (median 20 % for idle, about
   17 % otherwise). These are consistent with stale PID values re-logged between updates. They are
   kept and counted, not removed.
5. **Ranges (all files, after correct parsing):** RPM 0–3,769; speed 0–127 km/h; coolant 21–96 °C;
   control-module voltage 7.56–14.43 V (median 13.73); intake air 23–68 °C; MAP 16–102 kPa. No
   physically impossible or non-numeric values were found (0 in total). Voltage below 12 V occurs in
   101 files, always within the first 0–4 s of run time (engine start).
6. **Unused columns.** MIL/DTC-clearance counters, commanded actuator values and pedal positions are
   not mapped.

**Ground truth:** none. The file name encodes the operating mode only.
**Supported tasks:** recorded replay ✅, unlabelled anomaly detection ✅, supervised fault
classification ❌, injected-anomaly stress tests ✅ (clearly labelled synthetic).
**Split:** whole trips, stratified by category, seed `20261022` → train 77 / calibration 26 /
test 26 trips (`datasets/splits/toyota_carobd_split.json`). No trip spans partitions, so there is
no window overlap. **All trips come from one vehicle.** Results never generalize to other vehicles.
**Decision: ACCEPT** as the primary replay and anomaly-benchmark dataset.

## 3. DieselOBD

**Provenance.** https://github.com/AbouAbdallah-Lounis/OBD-Dataset, commit
`a9c96dc146aeeb62ef75d3b453c238f5379ccff5`. CC BY 4.0 per README. Cite Lounis, Mekki & Hadjab
(2025). The repo also contains a proprietary `FILE.cbf` (binary; its header string reads "Foxwell", though
the README names an ANCEL scanner; not parsed) and the authors' Python scripts. Those scripts were **not executed**.

**Files used:** `MASTER_TRAIN_X.xlsx` (118,477 × 26) and `MASTER_TEST_X.xlsx` (33,838 × 26).
Columns: 11 PIDs (`LOAD_PCT ECT MAP RPM VSS IAT MAF FRP BARO VPWR AAT`), 14 DTC indicator
columns (`P0000 P0562 P0113 …`) and `Mode` (values 0/1/2, meaning undocumented).

**Findings:**
1. **No `vehicle_id` and no time column**, although the README's example row shows `vehicle_id`.
   Replay therefore uses a labelled *synthetic playback clock*. No vehicle or trip id is invented.
2. **Units contradict the README.** The README lists °C and kPa. The data show ECT 72–241, IAT
   46–144, AAT 32–127, BARO 14.1–14.6 and MAP 2.8–37. These are only plausible as **°F and psi**
   (BARO 14.4 psi = 99 kPa, ECT 189 °F = 87 °C). The fuel-rail median of 6,183 psi ≈ 42.6 MPa fits a
   common-rail diesel. The mapping converts with an explicit `unit_inferred` flag that the UI and
   quality report surface. **MAF (0.00–0.26) and VSS units are unresolved and are not mapped.**
3. **Duplicates:** 90,657 of 118,477 training rows (76.5 %) and 26,280 of 33,838 test rows are exact
   duplicates.
4. **Labels are vehicle-level.** The DTC columns form exactly 7 distinct signatures, for example
   `P0403+P0404+P2015+P2009` (28,125 train rows) and `P0000` (29,674 train rows, the 3 fault-free
   vehicles merged). There is also one empty signature (4,391 rows). DTCs that always co-occur have
   identical counts.
5. **Leakage:** **all 7 signatures appear in both the train and the test file**, so the shipped split
   is not vehicle-disjoint.

**Consequence.** A classifier trained on these rows would learn to recognise *which vehicle* a row
came from, not a fault. With 5 faulty vehicles that each carry a unique DTC set, a leave-one-vehicle
-out evaluation of named faults is impossible (each class has one vehicle). **Decision: RESTRICTED
USE.** Used for import/unit-handling tests and as a bundled Excel sample. The fault classifier stays
disabled (`ai/classifier_optional.py`).

## 4. Scania APS

UCI dataset 421, zip downloaded 2026-10-22. Train: 60,000 rows (59,000 neg / 1,000 pos), 170
anonymized features, 8.3 % missing (`na`). Test: 16,000 rows (15,625 / 375). The description file
states **GNU GPL v3**, © 2016 Scania CV AB. The UCI web listing's license was not re-verified.
These are heavy-truck air-pressure-system features, not passenger-car OBD PIDs, and not time-ordered.
**Decision: RESTRICTED USE**, a standalone benchmark only. It is never mixed with car data. Not used
by the app.

## 5. KIT Automotive OBD-II Dataset (not obtained)

Landing page https://radar.kit.edu/radar/en/dataset/bCtGxdTklQlfQcAq, DOI 10.35097/1130,
published 2023-06-21, CC BY 4.0, archive ≈11.6 MB. The page describes 10 signals (ECT, MAP, RPM, VSS,
IAT, MAF, absolute throttle, ambient temperature, pedal positions D/E) logged with a PLX Kiwi 3 and
the OBD Auto Doctor app. File names encode `<date>_<brand>_<model>_<from>_<to>_<condition>`, where
the condition is a *road* condition (normal / frei / Stau), **not a mechanical fault**.

**Why not obtained:** the download button requires accepting the RADAR4KIT terms of use in the
browser. That is the user's decision, so it was not done automatically.
**Manual steps:** open the landing page → *Download Dataset* → accept the terms → extract all CSVs
into `data/raw/kit_obd/` → run `python experiments/audit_datasets.py`. The audit then lists files and
columns. The app's alias auto-mapping and manual mapping editor can import the files. No KIT-specific
mapping was written, because its real column names have not been seen.

## 6. Normalized schema

See `core/schema.py`. Canonical fields carry fixed units (rpm, km/h, °C, %, kPa, V, g/s). The time
column is `t_s`, and its meaning is set by `time_basis` ∈ {`absolute`, `relative_recorded`,
`synthetic_clock`}. Data origin ∈ {`recorded`, `synthetic`, `live_obd`}. Mappings live in
`datasets/column_mappings/*.json` (carOBD, DieselOBD, AutoGuard exports).

## 7. Which files were used for what

| Purpose | Files | Count |
|---|---|---|
| Isolation Forest fit + MOMENT input standardization | carOBD `train` partition | 77 trips, 83,644 s of complete features |
| Threshold calibration (both models) | carOBD `calibration` partition | 26 trips, 33,365 (IF) / 33,739 (MOMENT) scored seconds |
| Held-out evaluation | carOBD `test` partition | 26 trips, 31,527 s |
| Bundled demo samples | carOBD long10, ufpe7, idle3 (all `test`); DieselOBD first 3,000 test rows | 4 files |
| Supervised experiments | none (see §3, §4) | – |

## Reproduce
```
git clone --depth 1 https://github.com/eron93br/carOBD data/raw/eron93br_carOBD
git clone --depth 1 https://github.com/willkn/odb2 data/raw/willkn_odb2
git clone --depth 1 https://github.com/AbouAbdallah-Lounis/OBD-Dataset data/raw/AbouAbdallah-Lounis_OBD-Dataset
# Scania: download https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip, unzip to data/raw/scania_aps/
python experiments/audit_datasets.py
```
