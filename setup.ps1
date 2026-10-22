# AutoGuard AI - development setup (needs internet ONCE; the app itself runs offline).
# Usage (PowerShell, from the project folder):  powershell -ExecutionPolicy Bypass -File setup.ps1 [-Cpu]
param([switch]$Cpu)
$ErrorActionPreference = "Stop"

py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
if ($Cpu) {
    & $py -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
} else {
    & $py -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
}
& $py -m pip install -r requirements-dev.txt
& $py -m pip install --no-deps momentfm==0.1.4

# Pinned third-party model weights (MIT license) -> models/moment-1-small
& $py tools\download_moment.py --revision 411e288267f82cce86296dbe4d6c8bc533cc162f

# Raw datasets are NOT required to run the app (bundled samples + trained artifacts suffice).
# To reproduce the audit / training / evaluation, see docs/DATASET_AUDIT.md "Reproduce".
& $py -m pytest -q
