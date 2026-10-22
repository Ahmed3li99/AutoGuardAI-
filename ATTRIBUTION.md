# Third-party data, models and libraries

AutoGuard AI's own code was written for this graduation project. The following third-party
material is used. It was **not** created by the project team.

## Model weights
| Asset | Source | License | Pinned version |
|---|---|---|---|
| MOMENT-1-small (pretrained time-series foundation model, AutonLab, CMU) | https://huggingface.co/AutonLab/MOMENT-1-small | MIT (model card) | revision `411e288267f82cce86296dbe4d6c8bc533cc162f`, `model.safetensors` SHA-256 `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877` |
| `momentfm` inference code | https://github.com/moment-timeseries-foundation-model/moment | MIT | 0.1.4 |

The project did **not** train MOMENT. It only standardizes inputs and calibrates an alert threshold
on recorded data (see docs/MODEL_CARD.md). The Isolation Forest in `models/iforest/` was fitted by
this project on the carOBD training partition.

## Datasets (bundled samples in `data/sample/`)
| Dataset | Source | License / terms | Bundled |
|---|---|---|---|
| carOBD, Toyota Etios 2014 | https://github.com/eron93br/carOBD (mirror: https://github.com/willkn/odb2) | No LICENSE file. README: "open-source repository, free to everyone"; asks users to cite IEEE paper https://ieeexplore.ieee.org/abstract/document/8891367 | 3 unmodified trip CSVs (long10, ufpe7, idle3) |
| DieselOBD | https://github.com/AbouAbdallah-Lounis/OBD-Dataset | CC BY 4.0 (README). Cite: K. Lounis, H. Mekki, W. Hadjab, *DieselOBD: A Real-World Dataset of OBD-II Parameters and Diagnostic Trouble Codes for Diesel Vehicle Fault Diagnosis*, 2025. | First 3000 rows of MASTER_TEST_X.xlsx (**modified: truncated**) |
| APS Failure at Scania Trucks | https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks | Description file states GNU GPL v3, (c) 2016 Scania CV AB | Not bundled |
| KIT Automotive OBD-II Dataset | https://doi.org/10.35097/1130 | CC BY 4.0 (landing page); download requires accepting RADAR4KIT terms | Not obtained |

The carOBD terms are informal. Confirm with the author before redistributing the samples publicly
outside the graduation submission.

## Libraries
Python 3.11, PySide6/Qt (LGPLv3), pyqtgraph (MIT), pandas, NumPy, scikit-learn (BSD), PyTorch (BSD),
Hugging Face transformers / huggingface_hub / safetensors (Apache 2.0), ReportLab (BSD), Matplotlib,
openpyxl (MIT), joblib (BSD). Optional: python-OBD (GPLv2) for the untested future connector.
