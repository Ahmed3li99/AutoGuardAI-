# Evaluation (executed 2026-10-22)

Every number here was produced by the scripts named below. Raw output:
`data/processed/calibration_summary.json`, `data/processed/evaluation_results.json`,
`data/processed/moment_sanity_check.json`.

```
python experiments/audit_datasets.py
python experiments/calibrate_thresholds.py --device cuda      # fits IF on train, thresholds on calibration
python experiments/evaluate_anomalies.py --device cuda         # test partition only
python experiments/moment_sanity_check.py
```
Environment: Windows 11, Python 3.11.9, i7-13th gen (16 logical CPUs), RTX 4060 Laptop 8 GB,
torch 2.14.1+cu126, scikit-learn 1.9.1. Seeds: split 20261022, IF 1234, injection 7.

## Data and partitions
carOBD only (one Toyota Etios), whole-trip split stratified by operating mode:
train 77 trips / calibration 26 / **test 26 trips = 31,527 s (8.76 h)**. No trip, and therefore no
window, appears in two partitions. Thresholds come from calibration trips only. Test trips were not
touched before this evaluation.

## A. Unmodified held-out trips (no labels)
There are no fault labels, so these trips are *assumed* to be normal operation. The table shows how
often each detector would alert on such data. It is not an accuracy figure.

| detector | alerts | alerts / hour | share of seconds in alert | by category (alerts/h) |
|---|---|---|---|---|
| rules | 0 | 0.00 | 0.00 % | all 0 |
| Isolation Forest | 7 | 0.80 | 0.67 % | long 3.36, live 0.73, idle/drive/ufpe 0 |
| MOMENT-1-small | 12 | 1.37 | 0.47 % | long 6.05, live 1.10, idle/drive/ufpe 0 |

The fraction of scored seconds above the attention threshold on test trips was 0.67 % (IF) and
0.65 % (MOMENT). The calibration target was 0.5 %, so the thresholds transfer approximately to new
trips of the same car. The alerts concentrate in `long` trips. The test trip `long10` contains the
dataset's maximum speed (127 km/h). The cause was not investigated further, and no alert was treated
as a fault.

## B. Synthetic injection stress test (NOT real-world performance)
Each test trip was copied, and **one artificial pattern** was injected at a seeded random position
(`core/analytics/injection.py`). Detection = any alert overlapping the injected interval. AUROC/AP
use per-second scores against injected labels. 95 % CI = Wilson interval.

| injected pattern | rules | Isolation Forest | MOMENT-1-small |
|---|---|---|---|
| coolant ramp to 112 °C, 180 s (n=25) | **25/25** (CI 0.87–1.00), median delay 65 s | 0/25; AUROC 0.76 | 0/25; AUROC 0.64 |
| voltage drop to ~11.4 V, 60 s (n=26) | **26/26** (0.87–1.00), delay 0 s | 8/26 (0.17–0.50); AUROC 0.95 | **26/26** (0.87–1.00); AUROC 0.996, AP 0.90 |
| RPM frozen, 60 s (n=26) | 16/26 (0.43–0.78), delay 1 s | 0/26; AUROC 0.55 | 0/26; AUROC 0.49 |
| speed noise ×3σ, 30 s (n=26) | 0/26 (no rule exists) | 7/26 (0.14–0.46); AUROC 0.85 | 2/26 (0.02–0.24); AUROC 0.56 |

**Reading.** The rule baseline catches the patterns it was designed for. The overheat delay is the
time the ramp takes to cross 105 °C. The frozen-RPM rule misses cases where speed changed less than
15 km/h. MOMENT reliably flags large level shifts in normally flat signals (voltage) but not slow
drifts or frozen values. That is consistent with the sanity check: it does not reconstruct normal
dynamics, so its score acts like a deviation detector. The Isolation Forest ranks injected seconds
above normal ones (AUROC 0.76–0.95 for 3 of 4 patterns), but at the 0.5 % alert-rate threshold it
seldom crosses it. These are tests of the software on artificial patterns, with n ≈ 26 per cell.
**They say nothing about real mechanical faults.**

## C. Latency and memory (test partition, CUDA)
| detector | total on 8.76 h of data | ms per hour of data |
|---|---|---|
| rules | 0.23 s | 26 |
| Isolation Forest | 0.58 s | 66 |
| MOMENT (CUDA) | 0.90 s | 103 |

MOMENT on the longest test trip (2,883 s): CUDA peak allocation 285 MB; CPU-only run 0.59 s. The app
runs analysis in a background thread, so the UI stays responsive.

## D. What was not done
- No evaluation on real labelled faults (none exist in the obtained data).
- No cross-vehicle evaluation (one vehicle in the accepted dataset; KIT not obtained).
- No supervised classifier evaluation (rejected for leakage, see DATASET_AUDIT.md §3).
- No human review of the 19 ML alerts on unmodified test trips beyond the category breakdown above.
- Threshold sensitivity was not swept. Only the 99.5 / 99.95 % quantiles were evaluated.
