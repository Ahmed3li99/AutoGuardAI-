# Demo walkthrough (≈ 8 minutes, fully offline)

1. **Turn Wi-Fi off.** Start `dist\AutoGuardAI\AutoGuardAI.exe` (or `python run.py`). The status bar
   reads *Offline · no session*.
2. **Data Import & Quality** → *Bundled sample* → "Toyota Etios – long trip" → *Load sample*.
   Point out: mapping `toyota_carobd_v1` detected, time basis *Recorded relative time*, quality
   table with real counts, 0 invalid values, the unmapped columns.
3. Optional: load the DieselOBD excerpt. Note the **synthetic playback clock** warning and the
   **unit INFERRED** flags (°F/psi). Explain why there is no fault classifier (DATASET_AUDIT §3).
4. *Save session & open replay* → **Replay**: Play, 8×/16×, Pause/Resume, drag the slider, Stop.
   Plots show recorded timing. The banner says *Recorded Replay*, not live.
5. **AI & Anomalies** → *Run analysis*. Show the detector table (versions, CUDA/CPU, calibration
   status), the score plots with thresholds, then click an alert for the evidence window and its
   explanation. Note the fault-classifier row: *unavailable*.
6. **Overview**: signal cards (N/A where absent), per-signal status, Health Index explicitly N/A.
7. **Data Import** → *Load synthetic demo* (`overheat_and_low_voltage`) → save → run analysis. Rules
   flag coolant and voltage, everything is labelled **Synthetic Demo**, and the ML detectors say
   *uncalibrated for this data*.
8. **Events & Alerts**: filter by severity and detector type, export CSV. **Trips & History**:
   reopen the first session. **Reports**: export the PDF and show the provenance, quality, charts,
   timeline and limitations.
9. **Settings**: change the coolant threshold, re-run analysis. Show the OBD section: disabled,
   *NOT TESTED ON A REAL VEHICLE*.

# Likely faculty questions

**Did you train an AI model?** We trained no deep model. MOMENT-1-small is a third-party pretrained
model (MIT). We only standardize inputs and calibrate a threshold on recorded calibration trips. We
fitted a small Isolation Forest on training trips as a baseline.

**How accurate is it?** On real faults: unknown, because none of the obtained datasets has
trustworthy fault labels. We report alert rates on 26 held-out trips (0–1.4 alerts/h) and a
*synthetic* injection stress test with confidence intervals (EVALUATION.md). We do not present these
as real-world accuracy.

**Why is MOMENT not clearly better?** We measured it. In our setup its reconstruction is no better
than a mean predictor (normalized MSE ≈ 1.4), identically under two transformers versions. It is
excellent on voltage-drop injections but misses slow drifts. We report that instead of assuming the
foundation model wins.

**Why not use DieselOBD for fault classification?** Its labels are per vehicle, it has no vehicle or
time column, 76.5 % of training rows are duplicates, and all 7 label signatures appear in both train
and test. Any accuracy would measure vehicle identity.

**Does it work with my car?** It works with recorded CSV/XLSX/JSON files from any car via the
mapping editor. Thresholds were calibrated on one Toyota, so the UI marks scores on other data as
uncalibrated. Live OBD is future work.

**What happens if data are missing?** They stay missing (N/A), are counted in the quality report and
are masked by the models. Nothing is filled with plausible fake values. Short gaps (≤ 3 s) are
carried forward only on the models' 1 Hz grid, and the report says so.

**Where does the data go?** Nowhere. Everything is local SQLite, with user-controlled deletion.
