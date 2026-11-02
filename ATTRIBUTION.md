# Third-party data, models and libraries

AutoGuard AI's own code was written for this graduation project. The following third-party
material is used. It was **not** created by the project team.

## Model weights
| Asset | Source | License | Pinned version |
|---|---|---|---|
| MOMENT-1-small (pretrained time-series foundation model, AutonLab, CMU) | https://huggingface.co/AutonLab/MOMENT-1-small | MIT (model card) | revision `411e288267f82cce86296dbe4d6c8bc533cc162f`, `model.safetensors` SHA-256 `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877` |
| `momentfm` inference code | https://github.com/moment-timeseries-foundation-model/moment | MIT | 0.1.4 |

The project did **not** train MOMENT. The app uses the published weights. It standardizes inputs and
calibrates an alert threshold on recorded data. A separate fine-tuning experiment (not shipped) is
documented in docs/MODEL_CARD.md. The expected-vs-actual model (`models/normal_behavior/`) and the
Isolation Forest (`models/iforest/`) were built and fitted by this project on the carOBD training
partition.

## Datasets (bundled samples in `data/sample/`)
| Dataset | Source | License / terms | Bundled |
|---|---|---|---|
| carOBD, Toyota Etios 2014 | https://github.com/eron93br/carOBD (mirror: https://github.com/willkn/odb2) | No LICENSE file. README: "open-source repository, free to everyone"; asks users to cite IEEE paper https://ieeexplore.ieee.org/abstract/document/8891367 | 3 unmodified trip CSVs (long10, ufpe7, idle3) |
| DieselOBD | https://github.com/AbouAbdallah-Lounis/OBD-Dataset | CC BY 4.0 (README). Cite: K. Lounis, H. Mekki, W. Hadjab, *DieselOBD: A Real-World Dataset of OBD-II Parameters and Diagnostic Trouble Codes for Diesel Vehicle Fault Diagnosis*, 2025. | First 3000 rows of MASTER_TEST_X.xlsx (**modified: truncated**) |
| APS Failure at Scania Trucks | https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks | Description file states GNU GPL v3, (c) 2016 Scania CV AB | Not bundled |
| KIT Automotive OBD-II Dataset | https://doi.org/10.35097/1130 | CC BY 4.0 (landing page); download requires accepting RADAR4KIT terms | Not obtained |
| Vehicle Energy Dataset (VED), 383 cars, Ann Arbor 2017–18 | https://github.com/gsoh/VED | Apache License 2.0 (LICENSE file). Cite: G. Oh, D. J. LeBlanc, H. Peng, *Vehicle Energy Dataset (VED), A Large-scale Dataset for Vehicle Energy Consumption Research*, IEEE T-ITS 2020, doi:10.1109/TITS.2020.3035596 | Not bundled (used locally for cross-car validation) |
| APMonitor Automotive Monitoring (2014 Ford Focus, 2021 Chrysler Pacifica) | https://apmonitor.com/pds/index.php/Main/AutomotiveMonitoring | **No license stated** on the page | Not bundled; used locally for evaluation only. The two column mappings in `datasets/column_mappings/` are the project's own work |

The carOBD terms are informal. Confirm with the author before redistributing the samples publicly
outside the graduation submission.

## Libraries
Python 3.11, PySide6/Qt (LGPLv3), pyqtgraph (MIT), pandas, NumPy, scikit-learn (BSD), PyTorch (BSD),
Hugging Face transformers / huggingface_hub / safetensors (Apache 2.0), ReportLab (BSD), Matplotlib,
openpyxl (MIT), joblib (BSD), arabic-reshaper (MIT) and python-bidi (LGPLv3) for Arabic PDFs, and
python-OBD 0.7.3 (GPLv2) with pyserial (BSD) and Pint (BSD) for the read-only OBD-II connection.

python-OBD is GPLv2 and is bundled in the EXE. Before distributing the EXE beyond the graduation
submission, check the GPL obligations (e.g. offer the source code) or ship python-OBD separately.

Bundled fonts (`app/assets/fonts/`, from https://github.com/google/fonts, licence texts next to the files): IBM Plex Sans Arabic and IBM Plex Mono (© IBM, SIL Open Font License 1.1), Archivo (© The Archivo Project Authors, SIL Open Font License 1.1).

PDF reports use a font from the user's Windows Fonts folder (Arial/Tahoma/Segoe UI) at runtime. No
font file is redistributed.
