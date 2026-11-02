# Advanced analysis — implementation and validation

Open a trip, then **Results → Advanced analysis / النتائج → التحليل المتقدم**.
The workbench has six tabs: model selection/training and the five requested tools.
Long computations run on workers; results can be exported as JSON. No new service account,
cloud endpoint or dependency is required.

## Features

| Tool | Implemented behavior | Important limit |
|---|---|---|
| Sensor graph | Trainable NumPy message-passing neural network; graph learned on training trips; causal ten-second features; next-second residuals rank sensors; graph visualization | Associations are not causes or probabilities of faulty parts |
| Active diagnosis | Expected information gain divided by observed query cost; refresh priority; at most four supported read-only PIDs per cycle; optional live OBD checkbox; recorded replay comparison | Hypotheses and likelihoods are heuristic observation patterns; budget is an estimated soft budget, not a serial timeout guarantee |
| Audio/vibration/OBD | Imports 16-bit PCM WAV plus CSV (`t_s,ax,ay,az`, acceleration in m/s²); explicit clock offsets; one-second features conditioned on RPM/load; event timeline and selected-event WAV + readings CSV export | Requires synchronized recordings and a user-selected known-normal interval; does not identify named mechanical faults |
| Federated learning | Round model export, independent client training, weight-delta export, bounded weighted aggregation, mandatory recalibration and held-out evaluation | Manual file transport between trusted participants; no server, secure aggregation or differential privacy; updates can leak information |
| Possible futures | Forecasts at 10/30/60 seconds; 128 sampled joint residual trajectories; marginal 90% calibration bands; empirical threshold exceedance estimates and alert text at ≥50% | Not a 90% guarantee for the complete path; estimates can become unreliable on different cars or operating conditions |

## Training and measurement

`python experiments/train_advanced.py` trains bundled graph/forecast artifacts in `models/advanced/`.
Uses the existing Toyota partition: **77 training trips, 26 calibration trips, 26 held-out test trips**.
Sensor selection/scaling and graph structure do not use the test partition. Window sampling is capped
to bound CPU/memory: up to 16,000 GNN evaluation windows and 6,000 forecast evaluation windows.
Full metrics are in `data/processed/advanced_evaluation.json` and provenance is inside each NPZ.

For the trained bundled models, marginal test coverage across eight sensors and three horizons was
**87.47%–92.14%**, against a nominal 90% target. This is coverage, not fault-detection accuracy.
The GNN's next-second MAE **did not beat persistence on any of the eight sensors**; for RPM it was
61.95 RPM versus 41.64 RPM for persistence. Consequently the GNN remains a research tool and is
not promoted to the main diagnostic verdict. No labeled real-fault accuracy is claimed.

Cross-vehicle check (`data/processed/advanced_generalization.json`): the graph model raises about **103–106 unusual-event alerts per hour** on the normal Toyota test trips, **67/h** on the Ford and **15/h** on the Chrysler. That is far too many to show as findings. It stays a research view, outside the verdict.

Personal model training needs five approved trips for the selected car: at least three training,
one calibration and one test trip, ordered by saved session ID. Synthetic trips are rejected here.
Model compatibility requires the selected sensors and a regular one-second time grid. Forecasting
refuses missing current sensor values. Models use numeric NPZ plus JSON, never pickle for exchange.

## Federated workflow

1. Coordinator selects a graph model and exports a **new round** base NPZ.
2. Each participant selects that same round NPZ and their car's approved references. At least four
   trips are required; the newest two are withheld from local training for calibration and test.
3. Each client exports its update NPZ. This includes weight deltas, client/round identifiers,
   base fingerprint and a training-window count; it contains no telemetry rows.
4. Coordinator selects at least two updates from distinct clients and separate local calibration
   and test trips. Save the aggregated model to a different file, then select it in Models.
5. Start a new round when continuing training. Wrong-base, wrong-round, duplicate, nonfinite or
   malformed updates are rejected; norms and sample weights are bounded and recorded in an audit.

Use compatible vehicle populations. Keeping raw trips local does not establish update privacy,
poisoning resistance against malicious peers, or improved performance across different vehicles.

## Fusion workflow

Audio time is `WAV elapsed time + audio offset`; vibration time is `CSV t_s + vibration offset`.
Features align to OBD with nearest-match tolerance 0.6 s. Missing modalities remain missing.
Choose the end of a known-normal interval; its complete aligned seconds are split 70/30 into
training/calibration, requiring at least 30/20 seconds respectively. Only subsequent observations
are reported as events. WAV supports up to four hours; accelerometer import is limited to 256 MB.
Event export writes approximately ±5 seconds of audio and a CSV with trip and clip clocks,
OBD readings, audio/vibration features and deviation scores. Original audio cannot be overwritten.

Validation uses generated WAV/accelerometer signals and mocked/simulated OBD, not a real vehicle
with synchronized microphone/accelerometer hardware. Sensor mounting, clock drift and driving noise
still require field evaluation. Audio is imported explicitly; no microphone is opened automatically.

## Regression fixes

- Reanalysis revokes a normal reference when warning/consistency evidence appears; reassignment
  to a different car also revokes approval.
- What-if selects the model compatible with the trip's sensors.
- Scenario baseline is recalculated using the same configuration as the modified copy.
- Long dropouts remain detectable even when most of a signal is missing; initial absence is excluded.
- Selective OBD acquisition is marked so unrequested readings are not called sensor dropouts.
- Reports use the trip's car, not whichever car is currently selected.
- Deleted trips and superseded analysis jobs do not receive stale analysis results.
- Imports retain their captured car and settings; an older import cannot replace a newer selection.
- Live recording waits for the polling worker to stop before loading the completed recording.

## Reproduce validation

Run with the project's virtualenv. On Windows sandbox sessions, use a workspace temporary folder
because atomic rename in the redirected system temporary directory can be denied:

```powershell
New-Item -ItemType Directory -Force .test-tmp | Out-Null
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:QT_QPA_PLATFORM='offscreen'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv\Scripts\python.exe -m pytest -q --basetemp=.test-tmp/verification
.venv\Scripts\python.exe tools/advanced_walkthrough.py
```

Automated coverage includes independent predictions, causal windows, missing/nonfinite readings,
numeric model loading, two-client file exchange, clipping/replay rejection, PID scheduling,
stereo phase cancellation, fusion alignment/event clips, UI rendering, and the regression fixes.
Passing these checks is evidence for the tested cases, not a guarantee of zero bugs.

### Verified build, 2026-10-27

- Final full suite at that time: 115 passed. Re-run 2026-11-02 after the fixes below: see `docs/test_log.txt`.
- Offscreen workbench walkthrough: graph, forecast and active acquisition completed on a bundled
  recorded Toyota trip; Arabic charts were visually inspected.
- Executable archive parity: 17 compiled project modules match their tested source; both bundled
  model files match source SHA-256 hashes (`docs/advanced_bundle_check.json`).
- Frozen GUI startup smoke test exited successfully (code 0).
- Frozen end-to-end self-test also exited 0: analysis, SQLite round-trip, Arabic/English PDF
  reports and CSV export succeeded (`docs/advanced_frozen_selftest.json`).
- Fixed an executable-only dependency collision: PyInstaller picked up Poppler's version-suffixed
  `icuuc.dll` from PATH, shadowing Windows' unversioned ICU API required by Qt. The spec excludes
  that DLL; the distributed folder uses Windows' system ICU. Keep the complete `dist/AutoGuardAI`
  directory together when distributing the app.

### Review fixes, 2026-11-02

- The Results page read an undefined `open_advanced`. It failed whenever a trip was open, and the
  *Advanced analysis* button never appeared. Fixed and covered by
  `tests/test_ui_smoke.py::test_results_page_with_a_trip_shows_advanced_button`.
- The executable and `advanced_bundle_check.json` above predate the 2026-10-27 retraining. They were
  rebuilt and re-checked after these fixes; see `docs/OFFLINE_TEST.md`.
