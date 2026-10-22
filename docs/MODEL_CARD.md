# Model Card

AutoGuard AI uses three detectors. None diagnoses mechanical faults. The named-fault classifier is
**disabled** (see §4).

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
