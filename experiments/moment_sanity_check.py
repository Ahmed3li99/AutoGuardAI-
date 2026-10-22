"""Reconstruction sanity check for MOMENT checkpoints (zero-shot, unmasked).

Reports normalized MSE = MSE / variance per series: 1.0 equals predicting the
window mean, ~0 means faithful reconstruction. Run on simple synthetic shapes
and on a real carOBD window. Results -> data/processed/moment_sanity_check.json

Usage:  python experiments/moment_sanity_check.py [model_dir ...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from common import PROCESSED, ROOT, TOYOTA_DIR

os.environ["HF_HUB_OFFLINE"] = "1"
import torch  # noqa: E402
import transformers  # noqa: E402
from momentfm import MOMENTPipeline  # noqa: E402

from core.data_sources.files import RecordedFileSource  # noqa: E402
from core.preprocessing.resample import resample_1hz  # noqa: E402

SIGNALS = ["rpm", "speed_kmh", "coolant_temp_c", "battery_voltage"]


def inputs() -> dict[str, torch.Tensor]:
    tt = np.arange(512)
    syn = np.stack([np.sin(tt / 20), np.sin(tt / 5), tt / 512.0, np.sign(np.sin(tt / 30))])
    s = RecordedFileSource(TOYOTA_DIR / "long10.csv").read_session()
    d, _ = resample_1hz(s.data, s.signals)
    real = d[SIGNALS].iloc[600:1112].to_numpy().T
    return {"synthetic[sine20,sine5,ramp,square]": torch.tensor(syn[:, None, :], dtype=torch.float32),
            "carOBD long10 600-1111s " + str(SIGNALS): torch.tensor(real[:, None, :], dtype=torch.float32)}


def main() -> None:
    dirs = sys.argv[1:] or [str(ROOT / "models" / "moment-1-small")]
    out = {"transformers": transformers.__version__, "torch": torch.__version__, "results": {}}
    for md in dirs:
        m = MOMENTPipeline.from_pretrained(md, local_files_only=True, model_kwargs={"task_name": "reconstruction"})
        m.init()
        m.eval()
        res = {}
        with torch.no_grad():
            for name, z in inputs().items():
                r = m(x_enc=z, input_mask=torch.ones(z.shape[0], 512)).reconstruction
                nmse = (((r - z) ** 2).mean(dim=(1, 2)) / z.var(dim=2).squeeze()).numpy()
                res[name] = [round(float(v), 3) for v in nmse]
        out["results"][os.path.basename(md.rstrip("/\\"))] = res
        print(md, res)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "moment_sanity_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
