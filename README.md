# AutoGuard AI: Offline Vehicle Telemetry, Anomaly Detection & Diagnostics Prototype

Educational Windows desktop prototype (Computer Science graduation project). It imports **recorded**
OBD-II telemetry, validates it, replays it with plots, flags unusual patterns with rules, an
Isolation Forest and the pretrained **MOMENT-1-small** foundation model, stores everything in local
SQLite and exports PDF/CSV reports. **It runs fully offline. No car, OBD adapter, cloud or internet is
needed at runtime.**

> Not a diagnostic or safety system. Anomaly scores are statistical, not fault diagnoses. Real-world
> fault-detection performance is unknown because no trustworthy fault labels were available.

## Status (what is real, what is not)
| Feature | Status |
|---|---|
| CSV / XLSX / JSON import, mapping editor, unit conversion, quality report | ✅ implemented, tested |
| Recorded replay (play / pause / resume / seek / 1–16× / stop / restart) | ✅ implemented, tested |
| Synthetic demo scenario (clearly labelled everywhere) | ✅ |
| Rule baseline, Isolation Forest (fitted here), MOMENT-1-small (pretrained, threshold calibrated) | ✅ evaluated on 26 held-out trips (see [EVALUATION](docs/EVALUATION.md)) |
| Alerts with evidence and provenance, history, events, PDF/CSV reports | ✅ |
| Named fault classification | ❌ disabled. No dataset supports it defensibly ([why](docs/DATASET_AUDIT.md#3-dieselobd)) |
| Vehicle Health Index (0–100 %) | ❌ intentionally N/A (no ground truth) |
| Live OBD-II (python-OBD, read-only) | ⚠️ code + mocked tests only. **NOT TESTED ON A REAL VEHICLE** ([plan](docs/FUTURE_OBD_INTEGRATION.md)) |
| Windows EXE (PyInstaller onedir) | see [OFFLINE_TEST](docs/OFFLINE_TEST.md) |

## Quick start (development)
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1     # venv, deps, pinned MOMENT weights, tests (internet once)
.venv\Scripts\python run.py
```
Then: *Data Import & Quality* → *Load sample* → *Save session & open replay* → ▶ Play →
*AI & Anomalies* → *Run analysis* → *Reports*. Walkthrough: [docs/DEMO_AND_FAQ.md](docs/DEMO_AND_FAQ.md).

Tests: `.venv\Scripts\python -m pytest -q` (log: [docs/test_log.txt](docs/test_log.txt)).
Build EXE: `.venv\Scripts\pyinstaller packaging\autoguard.spec --noconfirm` → `dist\AutoGuardAI\AutoGuardAI.exe`.

## Data
| Dataset | Use | Decision |
|---|---|---|
| carOBD Toyota Etios 2014 (129 trips, 41.7 h) | replay, IF training, threshold calibration, held-out test, demo samples | ACCEPT |
| DieselOBD (8 diesel vehicles) | import/unit tests, Excel sample. Labels vehicle-level, train/test overlap | RESTRICTED |
| Scania APS | truck benchmark only, never mixed with car data | RESTRICTED (not used by app) |
| KIT OBD-II | download requires accepting KIT terms. Manual steps documented | PENDING |

Full evidence: [docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md). Attribution and licenses:
[ATTRIBUTION.md](ATTRIBUTION.md).

## Architecture
`VehicleDataSource` (file / synthetic / future OBD) → column mapping → normalization + quality →
`NormalizedSession` (provenance: origin + time basis) → replay engine / 1 Hz features → rules, IF,
MOMENT → alerts → SQLite → UI and PDF/CSV. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
app/        PySide6 UI (pages, state, theme)        core/      schema, mapping, sources, validation, replay
ai/         rules, IF, MOMENT, thresholds, pipeline  database/  SQLite migrations + repositories
reporting/  PDF + CSV                                 experiments/ audit, calibration, evaluation, sanity check
datasets/   mappings, split, manifest                 data/sample/ bundled offline demo files
models/     iforest/, moment-1-small/ (pinned)        tools/     weight download, GUI walkthrough
```

## Key results (executed, see EVALUATION.md)
- Held-out carOBD trips: rules 0 alerts/h, Isolation Forest 0.80/h, MOMENT 1.37/h.
- Synthetic injection stress test (not real-world): rules 25/25 overheat, 26/26 voltage drop;
  MOMENT 26/26 voltage drop (AUROC 0.996) but 0/25 overheat; IF best on speed noise (AUROC 0.85).
- MOMENT-1-small reconstructs inputs no better than a mean predictor in this setup (normalized MSE
  ≈ 1.4, identical under transformers 4.33.3 and 5.20.2). Documented in [MODEL_CARD](docs/MODEL_CARD.md).

## Known limitations
One vehicle in the accepted dataset, no fault labels, thresholds calibrated on that one car, MOMENT
weaker than expected, carOBD redistribution terms informal, OBD connector untested, large EXE because
of the CUDA build of PyTorch.

## Offline & privacy
The app never makes network calls. Hugging Face offline mode is forced, and weights load from
`models/`. Data stay in `%LOCALAPPDATA%\AutoGuardAI\autoguard.sqlite3`. History can be deleted in
*Trips & History*.
