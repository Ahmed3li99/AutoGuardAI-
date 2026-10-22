# Architecture

```
                 ┌────────────────────── VehicleDataSource (core/data_sources/base.py) ──────────────────────┐
                 │ RecordedFileSource (CSV/XLSX/JSON)   SyntheticDemoDataSource   OBDDataSource (optional,   │
                 │ files.py                             synthetic_demo.py         untested, feature flag)   │
                 └───────────────┬───────────────────────────────┬──────────────────────────┬──────────────┘
                                 │ raw table                     │                          │ Reading stream
                                 ▼                               │                          ▼
        ColumnMapping (core/mapping.py, datasets/column_mappings/*.json, alias auto-map, manual editor)
                                 ▼
        normalize() (core/preprocessing/normalize.py): time basis, units, range checks
                                 ▼
        NormalizedSession = DataFrame[t_s + canonical fields] + SessionMetadata (origin, time basis, provenance)
              │                          │                                   │
              ▼                          ▼                                   ▼
   QualityReport               ReplayEngine (core/playback)          resample_1hz → features
   (core/validation)           Qt-free; QTimer drives tick()          │
              │                          │                           ▼
              │                          │        ┌─────── ai/pipeline.run_analysis (worker thread) ───────┐
              │                          │        │ rules_baseline   isolation_forest   moment_inference   │
              │                          │        │ thresholds.calibrated → Alert(evidence, version,origin)│
              │                          │        └───────────────────────┬────────────────────────────────┘
              ▼                          ▼                                ▼
        ┌──────────────────── SQLite (database/): sessions, readings, analysis_runs, anomaly_scores, ────────┐
        │                     alerts, model_versions, vehicles, datasets, settings; versioned migrations     │
        └───────────────┬──────────────────────────────────────────────────────────────┬──────────────────────┘
                        ▼                                                              ▼
           PySide6 UI (app/): Overview, Import & Quality, Replay,              reporting/: PDF (ReportLab +
           AI & Anomalies, Events, History, Reports, Settings                  Matplotlib), CSV exports
```

## Key decisions
- **One contract for all sources.** Validation, replay, analysis, history and reports only see
  `NormalizedSession`. A future adapter plugs in as another `VehicleDataSource`.
- **Provenance travels with the data.** `DataOrigin` (recorded / synthetic / live_obd) and `TimeBasis`
  (absolute / relative_recorded / synthetic_clock) live in the metadata, every alert row, every CSV
  row, the PDF header and the UI banner. A SQL `CHECK` constraint rejects unknown origins.
- **Missing means missing.** Absent signals are absent columns. Impossible values become NaN and are
  counted. Gaps longer than 3 s stay NaN and are masked by models. The UI shows `N/A`.
- **No GUI freezes.** File reading, analysis and PDF generation run in `QThreadPool` workers. Replay
  ticks every 100 ms and plots keep at most 3,600 rows.
- **Offline by construction.** HF offline env vars are set at start-up. MOMENT loads with
  `local_files_only=True` from bundled files. There is no network code in the app.
- **Paths.** `core/paths.py` resolves bundled resources (`sys._MEIPASS` when frozen) separately from
  the writable user dir `%LOCALAPPDATA%\AutoGuardAI` (DB, logs). `AUTOGUARD_HOME` overrides it.
