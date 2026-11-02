# Evaluation (executed 2026-10-22)

Every number here was produced by the scripts below. Raw outputs are in `data/processed/`:
`calibration_summary.json`, `evaluation_results.json`, `moment_sanity_check.json`, `moment_adaptation.json`.

```
python experiments/audit_datasets.py
python experiments/calibrate_thresholds.py --device cuda      # IF fit on train; IF + MOMENT thresholds on calibration
python experiments/train_normal_behavior.py                   # expected-vs-actual: fit on train, thresholds on calibration
python experiments/adapt_moment.py --epochs 12                # optional MOMENT adaptation experiment
python experiments/evaluate_anomalies.py --device cuda        # TEST partition only
python experiments/moment_sanity_check.py
```
Environment: Windows 11, Python 3.11.9, i7 13th gen (16 logical CPUs), RTX 4060 Laptop 8 GB, torch
2.14.1+cu126, scikit-learn 1.9.1. Seeds: split 20261022, IF 1234, expected-vs-actual 2024, MOMENT
adaptation 11, injection 7.

## Data and partitions
carOBD only (one Toyota Etios). Whole-trip split, stratified by operating mode: train 77, calibration
26, **test 26 trips = 31,527 s (8.76 h)**. No trip, and so no window, is in two partitions. All
thresholds come from calibration trips only. The test trips were used only here.

## Detectors
| key | what | trained by this project? |
|---|---|---|
| rules | configurable limit rules | no (heuristics) |
| **expected_vs_actual** | per-signal gradient-boosted "normal behaviour" regressors. Alert when the actual reading departs from the expected one (robust z, 99.9 % calibration quantile per signal). Temperatures must also leave the car's normal warm range. | yes: 7 small regressors, under 1 min on CPU |
| isolation_forest | Isolation Forest on 33 window features | yes |
| moment | MOMENT-1-small, published weights, plain reconstruction error | no (threshold only) |
| moment_adapted | MOMENT-1-small fine-tuned 12 epochs with its own masked objective, masked-imputation error | fine-tuned (experiment) |

## A. Unmodified held-out trips (no labels, assumed normal) — re-run 2026-10-23
| detector | alerts | alerts / hour |
|---|---|---|
| rules | 0 | 0.00 |
| consistency | 0 | 0.00 |
| isolation_forest | 7 | 0.80 |
| moment | 12 | 1.37 |
| moment_adapted | 5 | 0.57 |
| expected_vs_actual | 5 | 0.57 |

## B. Synthetic injection stress test (NOT real-world performance)
One artificial pattern per copied test trip, at a seeded random position
(`core/analytics/injection.py`). AUROC/AP use per-second scores. 95 % CI = Wilson.

Two detection definitions are computed (`evaluate_anomalies.py`):
- *lenient*: any alert of that detector overlapping the injected interval.
- **strict**: an overlapping alert that names the **injected signal**, and that is **new**. No alert
  of the same detector on that signal overlapped the interval in the *unmodified* copy of the trip.

**Both definitions gave identical counts for every detector and pattern** (re-run 2026-10-23). The
table below therefore holds under the strict definition: no detection is credited to an alert on
another signal or to an alert that already existed before injection.

| injected pattern | rules | consistency | **expected_vs_actual** | isolation_forest | moment | moment_adapted |
|---|---|---|---|---|---|---|
| coolant ramp → 112 °C, hold, cool down; 180 s (n=25) | 25/25 [0.87–1.00], delay 66 s | 0/25 [0.00–0.13] | **25/25 [0.87–1.00], delay 67 s, AUROC 0.90** | 2/25, delay 50 s, AUROC 0.82 | 0/25, AUROC 0.62 | 0/25, AUROC 0.51 |
| voltage → ~11.4 V, 60 s (n=26) | 26/26 [0.87–1.00], delay 0 s | 0/26 [0.00–0.13] | **26/26 [0.87–1.00], delay 0 s, AUROC 1.00** | 6/26, delay 12 s, AUROC 0.95 | 26/26, delay 0 s, AUROC 1.00 | 4/26, delay 26 s, AUROC 0.77 |
| RPM frozen, 60 s (n=26) | 15/26 [0.39–0.74], delay 0 s | 0/26 [0.00–0.13] | **7/26 [0.14–0.46], delay 41 s, AUROC 0.64** | 0/26, AUROC 0.48 | 0/26, AUROC 0.47 | 0/26, AUROC 0.48 |
| speed noise ×3σ, 30 s (n=26) | 0/26 [0.00–0.13] | 17/26 [0.46–0.81], delay 25 s | **12/26 [0.29–0.65], delay 5 s, AUROC 0.89** | 3/26, delay 8 s, AUROC 0.84 | 1/26, delay 18 s, AUROC 0.55 | 1/26, delay 18 s, AUROC 0.71 |
| **sensor fault:** coolant +25 °C step, 60 s (n=26) | 22/26 [0.67–0.94], delay 0 s | 26/26 [0.87–1.00], delay 0 s | **26/26 [0.87–1.00], delay 0 s, AUROC 1.00** | 3/26, delay 5 s, AUROC 0.77 | 0/26, AUROC 0.81 | 0/26, AUROC 0.65 |
| **sensor fault:** intake-air sensor silent, 60 s (n=26) | 0/26 [0.00–0.13] | 26/26 [0.87–1.00], delay 0 s | **0/26 [0.00–0.13], AUROC 0.57** | 0/26 | 0/26 | 0/26 |

**Reading** (re-run 2026-10-23, after two sensor-fault patterns were added and the overheat
profile got a gradual cool-down; the extra patterns shift the seeded positions, so counts differ
slightly from the first run, e.g. frozen RPM 15/26 instead of 16/26 for the rules).
- The expected-vs-actual model is the best ML detector on every vehicle pattern and ties for the
  lowest false-alert rate.
- The **sensor-consistency check** raised **0 alerts on the 26 clean test trips** (and 0 on all 129
  real trips). It catches the sensor faults: step 26/26, dropout 26/26, speed noise 17/26 (only noise
  bursts faster than 45 km/h per s). It correctly stays silent on vehicle faults (overheat, voltage),
  which are not data inconsistencies.
- Together, rules + consistency + expected-vs-actual cover every pattern except frozen RPM during
  slow driving (rule 15/26; it needs ≥ 15 km/h speed change in 30 s).
- The overheat delay (~66 s) is the time the ramp needs to leave the normal range.
- Zero-shot MOMENT is very good on level shifts in flat signals (voltage) but misses slow drifts and
  frozen values.
- The **adapted MOMENT** reconstructs much better (masked nMSE on calibration windows 1.365 → 0.487)
  but **detects less**. It learned to impute the injected patterns too. It is therefore *not* used in
  the app; it stays a documented experiment.
- n ≈ 26 per cell. These are tests of the software on artificial patterns, **not evidence about real
  mechanical faults**.

## C. Latency
Expected-vs-actual originally computed its regressors twice (once for scores, once for alerts). It
now shares one prediction. On the 41-min sample trip: **0.155 s → 0.080 s** (mean of 5 runs each,
same machine, same session). On the full test partition (8.76 h), the re-run took 2.1 s for
expected-vs-actual. Other detectors: rules 1.5 s, IF 2.8 s, MOMENT (CUDA) 3.0 s, adapted MOMENT 6.5 s
(these totals were measured while other jobs shared the machine, so treat them as upper bounds).
MOMENT on CPU for the longest 48-min trip: 6.3 s; CUDA peak 430 MB. Analysis runs in a background
thread.

## D. Explanations
Findings are built from all detectors (`ai/explain.py`). Alerts on the same signal family that
overlap in time (±20 s) become one finding. Confidence is *high* when a limit rule fired, *medium*
when expected-vs-actual or ≥ 2 methods agree, and *low* for a single statistical model (collapsed in
the UI). On the demo (real trip + 3 injected faults), 23 raw alerts became 11 findings. The 3
injected faults rank first, and every finding outside the injected intervals is labelled as coming
from the real recorded part (tested in `tests/test_explain_nbm.py`).

## E. What was not done
- No evaluation on real labelled faults (none exist in the obtained data).
- Cross-vehicle evaluation only without fault labels (VED, Ford, Chrysler, §G); KIT not obtained.
- No supervised classifier (rejected for leakage, see DATASET_AUDIT.md §3).
- No live test on a real car (simulated adapter only).
- Threshold sensitivity not swept (one quantile per detector).

## F. "Learn your car" (executed 2026-10-23, final design: evidence gate + offset-only for
temperatures/voltage + driving requirement)
Scripts: `experiments/select_personal_settings.py` chooses the settings on **calibration** trips
only. `experiments/evaluate_personal.py` then measures once on **test** trips. Reference sets: 4
seeded random draws of k = 3 or 5 test trips, judged on the remaining test trips. Part B transforms
every trip into a **simulated** different car (RPM ×1.18, coolant +6 °C, voltage −0.35 V). It is
an artificial transformation of real data, not a second real car. Detection is strict (alert on the
injected signal, new vs the clean copy). Totals are summed over the 4 draws.

Re-run 2026-10-23 with the final setting (`NOISE_UP_TO = 3`, chosen on VED dev cars, §G). The
previous design's numbers are in brackets.

| | k | false alerts/h (mean, range) | overheat: model / rules+model | voltage drop: model |
|---|---|---|---|---|
| **A. Same car**, base model | – | 0.64 [0.63–0.64] | 88/88 · 88/88 | 92/92 |
| A. adapted (app) | 3 | **0.41** [0.13–0.64] (before: 0.64) | 51/88 · **88/88** | 92/92 |
| A. refit from scratch (v1) | 3 | 21.2 [10.0–41.1] | 64/88 · 88/88 | 74/92 |
| A. adapted (app) | 5 | **0.31** [0.14–0.68] (before: 0.59) | 81/81 · 81/81 | 84/84 |
| **B. Simulated other car**, base | – | 5.27 [4.81–5.63] | 90/90 · 90/90 | 92/92 |
| B. adapted (app) | 3 | **0.71** [0.13–1.30] (before: 0.83) | 42/90 · **90/90** | 92/92 |
| B. refit from scratch (v1) | 3 | 23.9 [5.8–40.6] | 66/90 · 90/90 | 69/92 |
| B. adapted (app) | 5 | **0.64** [0.14–1.70] (before: 0.85) | 57/80 · **80/80** | 84/84 |

Price of the wider margins on the Toyota: the model alone catches fewer injected faults of moderate size
(same car, k=3: noisy speed 39/92 vs 43/92 for the base model, frozen RPM 20/92 vs 28/92; overheat by the
model alone 51/88, but the limit rule still catches 88/88).

**Reading.**
- On the same car, the evidence gate leaves the expected values unchanged. Since the final setting
  the margins can only get wider, so false alerts fall slightly (0.64 → 0.41/h) and sensitivity drops
  slightly (see above).
- On a car that differs from the base car, adaptation cuts false alerts about 6× (5.2 → 0.83/h with
  3 trips, 0.85/h with 5).
  In one inspected example (learning trips live14 + long4, calibration drive12), the learned map
  recovered the hidden change: RPM factor 1.12 vs 1.18 true, warm coolant +5.5 °C vs +6 true, voltage
  −0.3 V vs −0.35 true.
- The adapted *model alone* catches fewer injected overheats on the other car. The overheat is an
  absolute 112 °C, and the other car already runs ~94 °C, so it is only +18 °C above *its* normal.
  The model judges against the car's own normal, while the limit rule (105 °C) judges absolute danger.
  Together, as in the app, they catch 100 %.
- **Design history.** An intermediate version fitted a slope for temperatures too. It learned
  meaningless maps (×0.5 + 49 °C for a +6 °C car) and gave 1.67 false alerts/h at k=3. Temperatures
  and voltage now get an offset only (learns +6.08 °C). The settings selection on calibration trips
  was re-run for this version (`personal_selection.json`) and still picks gate k=2. Version 1 refitted the regressors from the few trips: 4–24 false alerts/h. That
  is why the app adapts instead. Without the driving requirement, one 3-trip draw whose learning trips
  were both idle produced 20.4 false alerts/h (`personal_evaluation_without_driving_rule.json`). The
  app now refuses such sets (≥ 10 min above 20 km/h in the learning trips). 1 such draw was rejected
  and redrawn in part A.
- Part B is a controlled simulation. Real other cars are tested in §G — and they showed that this
  simulation was **too mild**: the first design (margins 0.8–1.2×) did not work on them.


## G. Other real cars: VED, Ford, Chrysler (executed 2026-10-23)

Until now every number came from **one** Toyota. Here the shipped models run, unchanged, on real
recordings of other cars (datasets: DATASET_AUDIT §5a, §5b). Scripts:
`experiments/prepare_ved.py`, `experiments/select_cross_car_settings.py`,
`experiments/evaluate_other_cars.py` → `data/processed/{cross_car_selection,other_cars_evaluation}.json`.
No real faults are known in these logs, so every model alert on them is counted as a false alert.

**Which model.** VED has only RPM and speed (no coolant, voltage, throttle, MAP). The app now picks the
model from the sensors in the file: **≥ 6 of the 8 inputs → full model, otherwise the *drivetrain*
model** (RPM + speed only, trained on the same Toyota train/calibration split).

### G1. The base models do not transfer to other cars

| Recording | base model | false alerts/h | median RPM error while moving |
|---|---|---|---|
| Toyota, held-out test trips (§A) | full | 0.57 | – |
| VED, 107 test cars, 507 h | drivetrain | **25.3** (no car below 1/h) | 329 rpm |
| Ford Focus, highway, 7.7 h | full / drivetrain | **54.0** / 20.5 | 397 / 341 rpm |
| Chrysler Pacifica, highway, 1.1 h | full / drivetrain | **57.6** / 11.7 | 1,138 / 1,156 rpm |

Why (measured): on other cars the actual-minus-expected spread is **2–6× wider** than on the Toyota
(e.g. RPM 205–330 rpm vs 133 rpm on VED cars), and the gearing differs (RPM/speed above 60 km/h:
Toyota 28.9, Ford 26.7, Chrysler 16.9). The Ford and Chrysler highway speeds (110–127 km/h) and the
Ford's altitude (baro ≈ 84 kPa → load 78 % vs 24 % expected) are outside what the Toyota saw.

**Consequence in the app (changed):** when the expected-vs-actual model is not calibrated for the
trip's car (no personal model yet), its findings are shown with **low confidence** and are **not part
of the verdict**. The verdict says so and points to *My cars — learn your car*.

### G2. "Learn your car" on real other cars

*Setting choice.* The first design (margins 0.8–1.2×) barely helped on VED cars (22.6 → 20.5 false
alerts/h on the dev cars). One new setting was added: every signal learns the car's own noise margin,
**from 1× up to `NOISE_UP_TO` × the base margin** (never narrower). It was chosen on **109 VED dev
cars** (307 h) with a rule written before the run: *the smallest candidate with ≤ 2 false alerts/h*.
(The Toyota calibration partition was checked at the same time: 0.52 → 0.29 false alerts/h.)

| NOISE_UP_TO (dev cars) | first design | 1.5 | 2 | **3 (chosen)** | 5 |
|---|---|---|---|---|---|
| false alerts/h, pooled | 20.5 | 7.0 | 2.7 | **1.37** | 1.34 |
| frozen RPM caught (model + consistency) | 29 % | 13 % | 6 % | 5 % | 5 % |

*Measured once on the 107 VED **test** cars* (reference = each car's earliest trips until the app's
readiness rules pass, 3–10 trips; judged on up to 40 later trips; 25 of 132 test cars did not have
enough data):

| VED test cars (507 h) | base drivetrain | learned (app) |
|---|---|---|
| false alerts/h, pooled | 25.3 | **1.10** |
| false alerts/h, median car / 90th percentile | 25.0 / 39.7 | **0.0 / 2.3** |
| cars below 1 alert/h | 0 of 107 | **90 of 107** |
| cars with fewer / more alerts than base | – | 106 / 0 |
| frozen RPM 60 s caught, model (+ consistency) | 27 % (27 %) | **4 % (4.5 %)** |
| RPM +600 step 60 s caught, model (+ consistency) | 41 % (41 %) | **8 % (8 %)** |
| noisy speed 30 s caught, model (+ consistency) | 70 % (99.8 %) | 12 % (**99.8 %**) |

*Ford and Chrysler.*
- **Ford, drivetrain model.** Learned from the separate 2.6 h file (`automotive.txt`, cut into 3
  parts), then judged on the 7.7 h highway file: **20.5 → 0.26** false alerts/h. It learned speed ≈
  −1.3 + 1.20 × base, i.e. different gearing.
- **Full model, within one recording.** Learned from the first 3 × 12 min and judged on the rest
  (optimistic: same day, same road):
  - Ford: **71.1 → 4.7/h** over 7.1 h.
  - Chrysler: **50.8 → 5.9/h** over only 0.5 h.
  - Chrysler RPM error: 1,094 → 475 rpm. It learned RPM ≈ 664 + 0.63 × base (9-speed gearbox).

**Honest reading.**
- Learning the car is **necessary**, not optional: without it the model is unusable on another car.
  With it, false alerts fall ~20× and become comparable to the Toyota (VED 1.1/h, Ford 0.26/h).
- **The price is sensitivity.** On a car whose RPM is hard to predict from speed (automatic gearboxes,
  different gear ratios), the RPM model can then only see very large deviations. A frozen or offset RPM
  sensor is caught only ~5–8 % of the time. Noisy speed is still caught 99.8 %, by the consistency
  check, not the model.
- The full model still raises 4.7–5.9 alerts/h after learning, even within the same recording. That is
  not good enough. More reference data per car, or a base model trained on many cars, is needed
  (future work).

### G3. The consistency check on real data

| VED car type | hours | alerts/h | which |
|---|---|---|---|
| ICE (264 cars) | 2,644 | **0.063** | speed jumps only |
| HEV (93) | 1,335 | 56.0 | "moving with 0 RPM" (55.96) |
| PHEV (24) | 593 | 51.3 | "moving with 0 RPM" |

- On real petrol cars it is very quiet.
- On hybrids, "moving with 0 RPM" is **normal**: the engine stops while driving. **Fixed:** the car
  profile now has an *engine type*, and for hybrid/electric profiles this check is off (a status note
  says so). For an unknown profile, the explanation now says "if your car is a hybrid this is normal".
- **Real glitches in the Ford highway file:** 19 events with values no healthy car produces (RPM >
  7,500, coolant > 140 °C, intake air > 90 °C, MAF > 300 g/s, baro > 110 kPa, catalyst > 1,100 °C).
  - The consistency check and the limit rules flagged **4 of 19**, the temperature jumps.
  - RPM, MAF and baro spikes were not covered: import keeps them because they are inside the encodable
    PID range. **Added since: see G4 (now 14 of 19).**
- `auto_warmup.txt` shows a constant ambient −40 °C (the PID's raw 0 = "no value"). Nothing flags this
  yet either.

### G4. Implausible-spike check for RPM, MAF and barometric pressure (added after G3)

**Limits come only from derivation data**: Toyota train + calibration trips (RPM, baro) and the 109 VED
dev petrol cars (RPM, MAF). Script: `experiments/derive_plausibility_limits.py`. The rule was fixed
before looking at any held-out data: **limit = 1.25 × the largest real value, and 1.25 × the largest
real 1-second change**. This is the same kind of margin as the earlier checks (speed 33 → 45 km/h/s).

| signal | largest real value | largest real 1-s change | value limit | change limit | source |
|---|---|---|---|---|---|
| RPM | 6,605 | 4,568 rpm | 8,256 | 5,710 rpm | VED dev, 4.9 M s (Toyota max 3,151) |
| MAF | 259.3 g/s | 236.5 g/s | 324 | 296 g/s | VED dev, 3.4 M s |
| baro | 103 kPa | 3 kPa | 128.75 | 3.75 kPa | Toyota, 118,512 s (VED has no baro) |

**Measured on held-out data only** (`experiments/evaluate_plausibility.py` →
`data/processed/plausibility_evaluation.json`):

| held-out data | hours | spike alerts |
|---|---|---|
| Toyota test trips | 8.8 | **0** |
| VED test petrol cars (132) | 1,257 | **0** |
| VED hybrids (93) / plug-ins (24) | 1,335 / 593 | **0 / 0** |
| Chrysler / Ford `automotive.txt` | 1.1 / 2.2 | **0 / 0** |
| Ford highway file (real glitches) | 7.7 | 15 (see below) |

Ford highway file, 19 glitch events marked in G3:

| | RPM | MAF | baro | coolant / IAT / catalyst | total |
|---|---|---|---|---|---|
| events | 4 | 2 | 5 | 2 / 1 / 5 | 19 |
| flagged by the spike check | **3** | **2** | **5** | – | 10 |
| flagged by all checks + rules (before → now) | 0 → 3 | 0 → 2 | 0 → 5 | 2 / 1 / 0 → 2 / 1 / 1 | **4 → 14** |

- The **5 other spike alerts** in that file are baro readings that jump by 4–8 kPa for 1–3 s and
  come straight back (e.g. 83 → 91 → 83). In one second that is roughly a 700 m altitude change, so
  they are almost certainly glitches too. They were below the coarse marker used in G3 (> 110 kPa).
  They are not counted above.
- **Missed RPM glitch:** 3,432 → 7,551 → 3,485 rpm. The +4,119 rpm jump is smaller than the largest
  real 1-second change in VED (4,568), and 7,551 is below the value limit. The limit was **not** moved
  to catch it, because that would be tuning on the evaluation file.
- **Still missed:** 4 catalyst-temperature spikes (no check for catalyst).
- Caveat: a car that really revs above 8,256 rpm (some sports engines) would be flagged; the text says
  the reading is probably a glitch, not a fault.


## H. More clean data, valid labels, frozen-sensor check (executed 2026-11-02)

**Goal:** higher numbers *without* leakage or overfitting. Three changes, each decided on training or
derivation data only. Afterwards everything was measured once on held-out data.

### H1. Clean multi-car training data (`experiments/build_clean_dataset.py`, `datasets/splits/multicar_split.json`)
- **Source:** VED **dev** petrol cars only, split again **by car**: 106 train cars and 26 calibration
  cars. The 132 VED **test** cars, the Toyota test trips, Ford and Chrysler are never read for training.
- **Cleaning** (data quality only, decided before any result): trip ≥ 300 s; RPM and speed in ≥ 90 % of
  the seconds; physically possible values; engine running ≥ 80 %; no glitch from the consistency checks;
  exact duplicates removed; at most 3 h per car.
- **Dropped:** 2,741 too-short trips, 142 with missing signals, 71 with glitches, and 4,189 over the
  per-car cap.
- **Kept:** 254 h from 106 cars for training and 66 h from 26 other cars for calibration, plus the
  Toyota train/calibration partitions.
- **Leakage checks (the script stops on failure):** 0 cars shared between train, calibration and test,
  and 0 training trips identical to any of the 8,321 test trips (fingerprint of the RPM + speed series).

### H2. RPM + speed model trained on many cars (`experiments/train_multicar_drivetrain.py`)
**Overfitting check:** the median RPM error is 143 rpm on the training cars and 142 rpm on cars the
regressors never saw. Speed: 5.0 vs 5.1 km/h. There is no gap, so it is not memorising cars. The
previous Toyota-only model is kept in `models/normal_behavior_drivetrain_toyota_only/` for comparison.

### H3. Frozen-reading check (`ai/consistency.py`, `experiments/derive_stuck_limits.py`)
The check flags a reading that repeats the exact same value. Limits are the shortest within ≤ 0.05
alerts/h on 1,420 h of derivation data:

| frozen reading | limit | rate on derivation data |
|---|---|---|
| RPM while driving and speed changing | 8 s | 0.012 alerts/h |
| non-zero RPM at any speed | 20 s | 0.048 alerts/h |
| speed while driving and RPM changing | 20 s | 0.039 alerts/h |

- The first rule, "1.25 × longest run", gave 132 s, which is useless. The longest runs were logger
  stalls in VED, so the rule was replaced before any held-out data was evaluated.
- RPM = 0 is left to the "engine off while moving" check, which has the hybrid exemption.

### H4. Valid labels (`core/analytics/injection.changes_data`)
An injected fault only counts if it really changed the signal in its window. For example, "freezing" a
reading that was already constant is not a fault. In this run all injections changed the data
(0 skipped), so no result was filtered by this rule.

### H5. Results on held-out data, measured once

**Toyota test trips** (26 trips, 8.8 h). "The app" = rules **or** consistency **or** the
expected-vs-actual model, i.e. what decides the on-screen verdict. Strict detection: the alert names the
injected signal and is new compared with the clean copy.

| injected fault | before | **now (the app)** |
|---|---|---|
| overheat | 25/25 | **25/25** |
| voltage drop | 26/26 | **26/26** |
| frozen RPM 60 s | 15/26 (58 %) | **26/26 (100 %)** |
| noisy speed 30 s | 17/26 | 17/26 (65 %) |
| coolant sensor step +25 °C | 26/26 | **26/26** |
| sensor silent | 26/26 | **26/26** |
| false alerts on normal trips | rules 0 · consistency 0 · model 0.57/h | **unchanged** |

**132 VED test cars** (107 had enough data; 507 h of later trips):

| | Toyota-only model | **multi-car model** |
|---|---|---|
| false alerts/h, car not learned yet | 25.3 | **0.73** (95 of 107 cars < 1/h) |
| false alerts/h, after "learn your car" | 1.10 | **0.20** (102 of 107 cars < 1/h) |
| median RPM error while moving | 329 rpm | **210 rpm** |
| frozen RPM 60 s caught (app) | 4.5 % before H3 | **99.6 %** (2,287/2,297) |
| noisy speed caught (app) | 99.8 % | **99.7 %** |
| RPM +600 step caught (model) | 41 % (with 25 false alerts/h) | **6 %** (2 % after learning) |

**Ford highway** (7.7 h), RPM + speed model, car not learned: 20.5 → **0.52** false alerts/h. After
learning: 0.26/h.

**Chrysler:** the RPM + speed model is **worse** (11.7 → 27.9/h). For this file the app uses the full
model (it has 8 of 8 signals), so this does not reach the user. It does show that the multi-car model is
not universally better.

**Consistency checks on all VED petrol cars** (dev + test, 2,644 h): 0.195 alerts/h in total. That is
frozen RPM 0.093, speed jump 0.063 and frozen speed 0.039.

### Honest reading
- The big gains are real and came from **more clean data** (other cars) and a **simple physical check**,
  not from tuning on test data.
- **Still weak:**
  - An RPM sensor that is *offset* (reads +600 rpm but still moves) is rarely caught on other cars. The
    multi-car model is calmer, so it is also less sensitive to moderate offsets.
  - Noisy speed on the Toyota is caught 65 % of the time.
- There are still **no real labelled mechanical faults**. All detection numbers are on injected faults.
