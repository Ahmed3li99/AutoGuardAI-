# Offline & Packaging Test Record

## Build (executed 2026-10-23, v0.3.0)
- PyInstaller 6.22.3, `packaging/autoguard.spec`, **onedir** → `dist\AutoGuardAI\AutoGuardAI.exe`.
- Bundle **4.43 GB**. Most of it is PyTorch with CUDA libraries. It includes the MOMENT weights, the
  expected-vs-actual model, the Isolation Forest, mappings, samples, python-OBD and the Arabic PDF
  libraries.
- A CPU-only PyTorch build would be much smaller but was **not built or tested**.

## Executed checks
| Check | How | Result |
|---|---|---|
| Unit / integration / UI tests | `pytest -v` (source) | **95 passed**, 0 failed, 0 skipped (`docs/test_log.txt`) |
| Frozen end-to-end self-test | `AutoGuardAI.exe --selftest <dir>`: sample → validation → rules + expected-vs-actual + IF + MOMENT → SQLite → reload → findings → PDF (AR + EN) + CSV. Also the demo with injected faults and the simulated adapter. | **exit 0**. All 4 detectors `ok` (MOMENT on `cuda`). 14 alerts saved = 14 reloaded. Verdict "Car looks OK — 6 observations". Demo: voltage + overheat as *warning*, frozen RPM found. Simulated adapter returned 10 signals. Both PDFs written. |
| v0.3 features in the frozen EXE | same self-test, extended | scenario lab: gradual voltage decline found by expected-vs-actual at 190 s, rules 293 s, IF 304 s. What-if removes only the voltage finding. Car adaptation learns +6.08 °C for a +6 °C copy. Consistency check `ok`. Report version 0.3.0. |
| Other-car features in the frozen EXE (rebuilt 2026-10-23) | same self-test, extended | An RPM + speed-only log uses the `drivetrain` model. A hybrid profile raises 0 "moving with 0 RPM" alerts. An RPM value of 11,596 is flagged as `consistency:rpm_spike`. All 5 detectors `ok`. Self-test 109 s, exit 0. GUI smoke test exit 0. Bundle 4.5 GB. |
| New brand/UI in the frozen EXE (rebuilt 2026-10-23) | self-test + `ExtractIconExW` on the EXE | The EXE carries the AutoGuard icon. The logo and line icons load inside the bundle (`brand_assets` all true). All 5 detectors `ok`, exit 0. GUI smoke test exit 0. |
| Rebuild after review fixes + multi-car model (2026-11-02) | `pytest` (136 passed), frozen self-test, GUI smoke, `tools/verify_advanced_bundle.py` | All passed. The EXE contains the multi-car RPM + speed model (byte-identical to source), the frozen-sensor check and the Results-page fix. 18 compiled modules match source. |
| No network use | `Get-NetTCPConnection -OwningProcess`, sampled every 0.3 s during the frozen self-test | **0 TCP connections** |
| Frozen GUI start | `AUTOGUARD_SMOKE_TEST=1 AutoGuardAI.exe` | **exit 0** |
| GUI flows (from source) | `tools/gui_walkthrough.py` in Arabic (demo) and English (real long trip). Screenshots of every screen reviewed. Live screen with the simulated adapter. | renders correctly, RTL in Arabic. Findings, charts, report and live values shown. |

Timing: the frozen self-test took 100 s in total, mostly start-up and CUDA library loading. The
analysis itself on a 41-min trip: rules 0.02 s, expected-vs-actual 0.16 s, IF 0.05 s, MOMENT 0.48 s.

## Bug found and fixed during this round
Loading a file in the background sometimes hung. The completion callback was a plain Python closure,
which PySide ran on the *worker* thread, so it touched widgets from there. Completion now goes through
bound methods of the GUI-thread `AppState`, which are queued to the GUI thread.

## Found while building v0.3 (fixed)
- The injected overheat ended with an instant 112 → 87 °C drop. The new consistency check (rightly) flagged it as an impossible jump. The profile now cools down gradually.
- The personal adaptation fitted a meaningless slope for temperatures (×0.5 + 49). Temperatures and voltage are now offset-only.
- The first "learn your car" design (refit from few trips) raised 11–24 false alerts/h. It was replaced; see EVALUATION §F.

## Bugs reported by the user's review (2026-10-23), fixed with regression tests
All in `tests/test_bug_regressions.py`:
1. **Analysis result saved to the wrong trip** when another trip was opened before it finished.
   Each job now carries its trip id and a generation number. The result is saved to *its* trip and
   shown only if that trip is still open (reproduced with an artificially delayed analysis).
2. **"No issues found" while models were unavailable.** The verdict now lists the checks that ran
   and says "partial check" when any did not.
3. **A short trip hid a rule warning** (120 °C for 30 s showed "not enough data"). Warnings now take
   priority, with a note that the models did not run.
4. **OBD time ran backwards after a reconnect** (`[10, 1]`). The session clock now starts once; a
   reconnect is logged and shows as a gap (`[10, 11]`).
5. **A missing time value hid unsorted rows** (`[2, NaN, 1]` → duration −1 s). Missing times are now
   dropped before checking order (`[1, 2]`).

## NOT executed (do these before the defence)
- [ ] Run with Wi-Fi physically off (the 0-connection measurement is strong evidence but not the same test).
- [ ] Click through every control in the **frozen** EXE by hand.
- [ ] A PC without an NVIDIA GPU (CPU fallback is tested from source).
- [ ] A clean Windows PC without Python.
- [ ] **A real car with a real ELM327 adapter** (see FUTURE_OBD_INTEGRATION.md).

## Manual offline checklist
1. Copy `dist\AutoGuardAI\` to the target PC. Turn Wi-Fi off.
2. Start `AutoGuardAI.exe` → Home. Open the sample long trip → Results appear automatically.
3. Run demo → red verdict with 2 important issues → *Show on chart* → Trip screen.
4. Report → Arabic PDF → opens with readable Arabic text.
5. Connect to the car → *simulated adapter* → values update → record 30 s → *Stop & analyse*.
6. 🌐 language switch both ways. Settings opens.
7. Optional: `AutoGuardAI.exe --selftest C:\temp\ag` → `selftest.json` has `"ok": true`.
