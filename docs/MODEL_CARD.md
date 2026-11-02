# Model Card

AutoGuard AI uses four detectors. None diagnoses mechanical faults. The named-fault classifier is
**disabled** (see §4). The **primary** ML detector is the expected-vs-actual model (§0), chosen
because it measured best on the held-out evaluation (docs/EVALUATION.md) and because every alert
carries an interpretable expected value.

## 0. Expected-vs-actual ("normal behaviour") model (`ai/normal_behavior.py`)
- **Built and trained by this project** on the carOBD train partition (77 trips), with thresholds from
  the calibration partition (26 trips). Version `expected-vs-actual-carobd-v1-2026-10-22`.
- One `HistGradientBoostingRegressor` (300 iterations, lr 0.08, 31 leaves, seed 2024) per target:
  coolant, RPM, control-module voltage, MAP, load, intake-air temperature, speed.
- Inputs: the *other* signals, engine run time (capped at 1 h), 60 s / 300 s rolling means of load,
  speed and RPM, and acceleration. The target's own value is **never** an input, so a frozen or
  drifting sensor cannot explain itself.
- Score: |actual − expected| / (1.4826·MAD of calibration residuals), with a 5 s rolling median.
  Alert when it exceeds the 99.9 % calibration quantile of that target for ≥ 8 s.
- Temperatures (coolant, intake air) must also leave the car's normal warm range (2.5–97.5 %
  percentile of warm-engine training data). This prevents false alarms on warm restarts, where the
  model would otherwise "expect" a cold engine.
- Each alert stores actual value, expected value and the typical range. These feed the plain-language
  explanation (`ai/explain.py`).
- Artifact `models/normal_behavior/nbm.joblib`, SHA-256-checked before loading (`ai/artifacts.py`).

## 0a. "Learn your car": personal adaptation (`ai/personal.py`)
The user approves **reference trips** (normal driving). A trip with a confirmed rule warning cannot
be approved, so a fault is never learned as "normal". The base expected-vs-actual model is then
**adapted** to the car, **not refitted**:

- **Evidence gate:** a signal is adapted only if its median offset from the base expectation is at
  least 2× the base noise scale. A car like the base car therefore changes (almost) nothing.
- **Learned map:** for gated signals, *this car ≈ a + b × base car* (Huber regression on the older
  reference trips, b clipped to 0.5–2). Coolant, intake-air temperature and voltage get an **offset
  only** (b = 1). A thermostat or alternator set-point shifts the level; a slope fitted across a cold
  start was meaningless (×0.5 + 49 for a +6 °C car; with offset-only it learns +6.08 °C). Inputs are
  mapped into the base scale before prediction and mapped back afterwards.
- **Scale and threshold:** the residual scale is re-estimated on the newest reference trips (never
  the same trips). Adapted signals are clipped to 0.8–3× the base scale. All other signals are clipped
  to 1–3×, so they only ever become less sensitive (`NOISE_UP_TO = 3`, chosen on 109 VED dev cars; see
  EVALUATION §G). The z threshold stays the base model's. Typical ranges are the base ranges mapped
  through the learned map.
- **Measured on real other cars (EVALUATION §G):** 107 VED test cars 25.3 → 1.1 false alerts/h; Ford
  20.5 → 0.26/h. The price: an RPM sensor that freezes or shifts is then caught by the model only
  ~5–8 % of the time on those cars.
- **Readiness rule:** ≥ 3 reference trips, ≥ 30 min, and the learning trips must include ≥ 10 min of
  driving above 20 km/h. Learning only from idle trips gave 20 false alerts/h in testing. 5 trips are
  recommended.
- **Design history (measured, `docs/EVALUATION.md` §F):** refitting regressors from 3–5 trips gave
  4–24 false alerts/h. Ungated adaptation added false alerts on the same car. Settings were chosen on
  the calibration partition (`experiments/select_personal_settings.py`) and then measured once on
  test trips.
- The learned map is shown to the user in words (e.g. "engine runs 5.5 °C hotter, ~13 % higher RPM
  at the same speed").

## 0a2. Which expected-vs-actual model runs
- **Full model:** used when the log has ≥ 6 of its 8 inputs.
- **Drivetrain model** (`models/normal_behavior_drivetrain`, RPM + speed only). Since 2026-11-02 it is
  trained on the Toyota train trips plus 254 h of clean data from 106 VED cars, and calibrated on 66 h
  from 26 other cars (split by car, leakage-checked, EVALUATION §H). There is no overfitting gap: the
  error is 143 rpm on training cars and 142 rpm on unseen cars. The previous Toyota-only version:
  used otherwise, e.g. for VED-style logs.
- A personal model is used only for trips of its car, and only when it was learned for the same
  variant.
- On a car that has not been learned, the model's findings are **low confidence** and do not enter
  the verdict. On other real cars the base model raised 12–58 false alerts/h (EVALUATION §G1).

## 0b. Sensor-consistency checks (`ai/consistency.py`, version `consistency-v1`)
"Is the car sick, or is the sensor lying?" These checks judge the **data**, not the car. Each one
flags a pattern a real car can hardly produce. Thresholds were set from the largest 1-second
changes in **all 129 real carOBD trips** (150,118 s at 1 Hz):

| check | real data max | flagged above |
|---|---|---|
| coolant jump | 3 °C/s | 8 °C/s |
| intake-air jump | 3 °C/s | 10 °C/s |
| speed jump | 33 km/h per s | 45 km/h per s (> 1.25 g) |
| moving > 20 km/h with RPM = 0 | never (0 of 150,118 s) | ≥ 3 s |
| one sensor silent while the others report | – | ≥ 10 s |
| RPM value / 1-s change (consistency-v2) | 6,605 / 4,568 rpm (VED dev cars) | 8,256 / 5,710 |
| MAF value / 1-s change | 259 / 236 g/s (VED dev cars) | 324 / 296 |
| barometric pressure value / 1-s change | 103 / 3 kPa (carOBD train+calibration) | 128.75 / 3.75 |
| frozen RPM while driving (consistency-v3) | 0.012 alerts/h at 8 s (1,420 h derivation data) | ≥ 8 s |
| frozen non-zero RPM, any speed | 0.048 alerts/h at 20 s | ≥ 20 s |
| frozen speed while driving | 0.039 alerts/h at 20 s | ≥ 20 s |

Result on all 129 real trips: **0 alerts**. On other real cars (VED, never used to set thresholds):
0.063 alerts/h on 2,644 h of 264 petrol cars. On hybrids "moving with 0 RPM" is normal (56/h on 93
HEVs), so it is switched off when the car profile's engine type is hybrid/electric. On a real Ford log
it flagged 4 of 19 glitch events. With the spike check for RPM/MAF/baro (limits 1.25× the largest real
value / 1-s change in derivation data only): **0 alerts on 3,195 h of held-out real driving** and 14 of
19 Ford glitch events flagged (EVALUATION §G4). Findings involving these checks (plus the frozen-RPM rule
and speed readings that contradict RPM/load in the expected-vs-actual model) are explained as **data
reliability** ("likely a measurement or recording problem"), with the other sensors' values at that
time as evidence. If a limit rule also fired, the finding stays a warning but is marked as a doubtful
reading ("if the dashboard gauge agrees, treat it as real").

## 1. Rule baseline (`ai/rules_baseline.py`, version `rules-v1`)
Configurable heuristic thresholds on the 1 Hz grid, each with a minimum duration. Defaults:
coolant ≥ 105 °C attention / ≥ 110 °C warning (≥ 10 s); control-module voltage < 12.6 V attention /
< 12.0 V warning while RPM > 500 (≥ 10 s); > 15 V warning; RPM > 4,500 attention (≥ 3 s); RPM
frozen for 30 s while speed changed ≥ 15 km/h. These are generic passenger-car heuristics. They
**are not manufacturer specifications**, and the UI lets the user change them.

## 2. Isolation Forest baseline (`ai/isolation_forest_baseline.py`)
- **Fitted by this project** on the carOBD train partition (77 trips, 83,644 complete feature rows).
- Features per second for 8 signals (rpm, speed, coolant, load, throttle, MAP, IAT, voltage): value,
  30 s rolling mean, rolling std and first difference, plus acceleration from speed and time
  (33 features). StandardScaler + IsolationForest (200 trees, max_samples 4096, seed 1234).
- Score = −`score_samples` (higher = more unusual).
- Artifact `models/iforest/iforest.joblib`. Its SHA-256 is recorded in `iforest_meta.json`, and
  loading refuses a mismatching file because joblib can execute code.
- Version `iforest-carobd-v1-2026-10-22`, scikit-learn 1.9.1.

## 3. MOMENT-1-small (`ai/moment_inference.py`)
| | |
|---|---|
| Source | AutonLab/MOMENT-1-small, https://huggingface.co/AutonLab/MOMENT-1-small, MIT license |
| Revision | `411e288267f82cce86296dbe4d6c8bc533cc162f` |
| Weights | `model.safetensors`, 151,615,328 bytes, SHA-256 `785e6c6f…aa877` (full hash in PROVENANCE.json). The pickle `pytorch_model.bin` is deliberately not shipped. |
| Architecture | T5 encoder (8 layers, d_model 512, 35.3 M parameters), patch length 8, input length 512 |
| Code | `momentfm` 0.1.4 (installed `--no-deps`), PyTorch 2.14.1+cu126, transformers 5.20.2 |
| Training by this project | **None.** The backbone and head are used as published. |
| Project-specific parts | Per-signal mean/std from the carOBD train partition; attention/warning thresholds = 99.5 / 99.95 % quantiles of calibration-partition scores |
| Offline | `HF_HUB_OFFLINE=1`, `local_files_only=True`; weights loaded from `models/moment-1-small/` |

**Scoring.** 1 Hz grid → standardize 8 signals → 512 s windows with stride 256 (short sessions
left-padded and masked; gaps masked) → deterministic `reconstruct()` → squared error per channel and
second, overlaps averaged → mean over channels. The per-channel errors are shown as "largest
deviation in".

**Feasibility verification (executed).**
- All 80 tensors in the checkpoint load and are bit-identical to the safetensors file.
- Output is deterministic. CUDA ≈ 34 ms per batch of 4 windows × 8 channels; CPU ≈ 215 ms.
- Outputs under transformers 4.33.3 (the version momentfm pins) and 5.20.2 differ by at most 6e-7.
- **Reconstruction sanity check** (`experiments/moment_sanity_check.py`, results in
  `data/processed/moment_sanity_check.json`): normalized MSE (1.0 = predicting the window mean) is
  1.33–1.43 on simple sines/ramp/square and 1.40–1.48 on a real carOBD window. MOMENT-1-base,
  downloaded for comparison only, scored 1.08–1.20. Toggling the positional embedding did not help.
  The inference call matches the official anomaly-detection tutorial, which uses MOMENT-1-**large**.

**Interpretation.** In this setup MOMENT-small does not reproduce its input. Its error therefore
behaves mostly like a per-window deviation from typical levels, not like a learned model of normal
dynamics. This matches the evaluation: it detects injected voltage drops very well (a large level
shift in a normally flat signal) but not slow overheating, frozen RPM or speed noise. It is kept as
an *evaluated candidate* that performs as reported in `docs/EVALUATION.md`, not as a superior model.

**Adaptation experiment** (`experiments/adapt_moment.py`, results in
`data/processed/moment_adaptation.json`). Starting from the published weights (not from scratch), the
whole model was fine-tuned for 12 epochs with its own masked-patch objective (mask ratio 0.3, AdamW,
lr 1e-4) on 398 train windows (512 s × 8 signals). It took 188 s on the RTX 4060, and the best epoch
was selected on 92 calibration windows. Masked-patch nMSE on calibration windows: **1.365 zero-shot →
0.487 adapted**. On the held-out stress test, however, the adapted model detected *fewer* injected
faults (e.g. voltage drop 3/26 vs 26/26), because it learned to impute those patterns as well. The app
therefore uses the published weights. The adapted weights are not shipped
(`models/moment-1-small-carobd/` is gitignored).

## 4. Fault classifier: disabled
The UI shows "Specific fault classification unavailable for this dataset/model". Reasons are in
`docs/DATASET_AUDIT.md` §3–4: DieselOBD labels are vehicle-level with a vehicle-overlapping
train/test split, and Scania APS is unrelated truck data. No open, documented, schema-compatible
pretrained car fault classifier was identified.

## 5. Intended use and limitations
- Educational prototype for *recorded* data. Not a safety system, not a diagnostic tool.
- All learned parts were calibrated on **one** Toyota Etios. On other vehicles the score scale is
  shown as "uncalibrated for this data".
- No real fault labels exist, so real-world detection performance is **unknown**.
- Displayed scores are uncalibrated anomaly scores, not probabilities.
