# Offline & Packaging Test Record

## Build (executed 2026-10-22)
- PyInstaller 6.22.3, `packaging/autoguard.spec`, **onedir**. Output `dist\AutoGuardAI\AutoGuardAI.exe`.
- Bundle size **4.42 GB**, of which PyTorch with CUDA 12.6 libraries is 3.9 GB and PySide6 103 MB.
  It includes the MOMENT weights (152 MB), the Isolation Forest, mappings and the 4 sample files.
- A CPU-only PyTorch build (`setup.ps1 -Cpu`) should shrink the bundle substantially, but this was
  **not built or tested**. `onefile` was not attempted.

## Executed checks
| Check | How | Result |
|---|---|---|
| Unit/integration tests | `pytest -v` (source) | **51 passed**, 0 failed, 0 skipped (`docs/test_log.txt`) |
| Frozen end-to-end self-test | `AutoGuardAI.exe --selftest <dir>`: bundled sample → validation → rules + IF + MOMENT → SQLite → reload → PDF + CSV | **exit 0**. Quality OK, 5,035 rows, all 3 detectors `ok`, MOMENT on `cuda`, 13 alerts saved = 13 reloaded, PDF 302 KB |
| No network use during analysis | `Get-NetTCPConnection -OwningProcess` sampled every 0.3 s during the frozen self-test | **0 TCP connections** |
| HF offline enforcement | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` set before imports; `local_files_only=True` | MOMENT loaded from `_internal\models\moment-1-small` |
| Frozen GUI start | `AUTOGUARD_SMOKE_TEST=1 AutoGuardAI.exe` (opens main window, quits after 1.5 s) | **exit 0** |
| GUI demo flow (from source) | `tools/gui_walkthrough.py` on the synthetic demo and on `toyota_carobd_long10.csv`: screenshots of all 8 pages reviewed | renders real data. Detectors ok. 13 alerts on long10 |
| Same results source vs EXE | alert count on long10 | 13 = 13 |

Timing: the frozen self-test took 141 s on its first (cold) run, mostly process start-up and
CUDA/PyTorch library loading. Analysis itself: rules 0.08 s, IF 0.27 s, MOMENT 1.87 s.

## NOT executed (do these before the defence)
- [ ] **Run with Wi-Fi physically disabled / Ethernet unplugged.** Changing network settings was
      outside what this session was allowed to do. The 0-connection measurement above is strong
      evidence but not the same test.
- [ ] Manual click-through of every control in the **frozen** EXE (only start-up and the headless
      pipeline were verified frozen; the full GUI flow was verified from source).
- [ ] A machine **without** an NVIDIA GPU (CPU fallback was tested from source: `test_moment_loads_offline_and_scores_cpu`).
- [ ] A clean Windows machine without Python installed.

## Manual offline checklist
1. Copy `dist\AutoGuardAI\` to the target PC. Disable Wi-Fi and unplug Ethernet.
2. Start `AutoGuardAI.exe`. The status bar shows *Offline · no session*.
3. Import → *Load sample* (long trip) → validation OK → *Save session & open replay* → Play 16×.
4. AI & Anomalies → *Run analysis*. All detectors `ok` (`moment` device `cpu` on non-NVIDIA PCs).
5. Click an alert → evidence window. Trips & History → reopen. Reports → export the PDF and open it.
6. Settings → OBD section reads *disabled / not connected*.
7. Optional headless check: `AutoGuardAI.exe --selftest C:\temp\ag` → `selftest.json` has `"ok": true`.
