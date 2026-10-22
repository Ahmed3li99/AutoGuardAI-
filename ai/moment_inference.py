"""Zero-shot anomaly scoring with the pretrained MOMENT-1-small foundation model.

Third-party model: AutonLab/MOMENT-1-small (MIT license), loaded ONLY from a
local directory with Hugging Face offline mode forced on. The backbone is
never trained here.

Signal/window -> score strategy
  1. 1 Hz grid; each model signal standardized with mean/std from the TRAIN
     partition (stored in moment_meta.json) so errors are comparable across
     channels.
  2. Windows of 512 s, stride 256 s (last window right-aligned). Shorter
     sessions are left-padded; padding and remaining gaps get input_mask=0
     (MOMENT ignores patches that are not fully observed).
  3. Deterministic reconstruction (``MOMENT.reconstruct``: no random masking),
     squared error per channel and second; overlapping windows averaged.
  4. Per-second score = mean squared error over channels observed that second.
     Channel errors are kept as the evidence for "largest deviation in".
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ai.thresholds import Threshold
from core.paths import MODELS_DIR

SEQ_LEN = 512
STRIDE = 256
MODEL_DIR = MODELS_DIR / "moment-1-small"


class MomentUnavailable(RuntimeError):
    pass


def _force_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@dataclass
class MomentScorer:
    model: object
    device: str
    signals: list[str]
    mean: np.ndarray
    std: np.ndarray
    meta: dict
    threshold: Threshold | None = None

    @property
    def version(self) -> str:
        return self.meta.get("version", "moment-1-small")

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR, device: str = "auto") -> "MomentScorer":
        meta_path = model_dir / "moment_meta.json"
        if not (model_dir / "config.json").exists():
            raise MomentUnavailable(f"MOMENT weights not found in {model_dir}. "
                                    "Run tools/download_moment.py during setup (needs internet once).")
        if not meta_path.exists():
            raise MomentUnavailable("moment_meta.json missing - run experiments/calibrate_thresholds.py.")
        _force_offline()
        try:
            import torch
            from momentfm import MOMENTPipeline
        except ImportError as e:
            raise MomentUnavailable(f"PyTorch/momentfm not installed: {e}") from e
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = MOMENTPipeline.from_pretrained(str(model_dir), local_files_only=True,
                                                   model_kwargs={"task_name": "reconstruction"})
            model.init()
            model.to(device).eval()
        except Exception as e:  # corrupt weights, device failure, version mismatch
            raise MomentUnavailable(f"Could not load MOMENT on {device}: {type(e).__name__}: {e}") from e
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        thr = Threshold(**meta["threshold"]) if meta.get("threshold") else None
        return cls(model, device, meta["signals"], np.array(meta["mean"]), np.array(meta["std"]), meta, thr)

    def missing_signals(self, available: list[str]) -> list[str]:
        return [s for s in self.signals if s not in available]

    def score(self, df_1hz: pd.DataFrame, batch_size: int = 16) -> tuple[np.ndarray, pd.DataFrame, dict]:
        """Return (per-second score, per-second per-channel error, timing info)."""
        import torch

        miss = self.missing_signals(list(df_1hz.columns))
        if miss:
            raise ValueError(f"MOMENT scorer needs signals not present: {miss}")
        x = df_1hz[self.signals].to_numpy(dtype=np.float32)          # T x C
        x = (x - self.mean) / self.std
        obs = np.isfinite(x)                                         # T x C
        T, C = x.shape
        x = np.where(obs, x, 0.0).astype(np.float32)

        if T >= SEQ_LEN:
            starts = list(range(0, T - SEQ_LEN + 1, STRIDE))
            if starts[-1] != T - SEQ_LEN:
                starts.append(T - SEQ_LEN)
            pad = 0
        else:
            starts, pad = [0], SEQ_LEN - T
        err_sum = np.zeros((T, C), dtype=np.float64)
        err_cnt = np.zeros((T, C), dtype=np.float64)
        t0 = time.perf_counter()
        with torch.no_grad():
            for b in range(0, len(starts), batch_size):
                batch_starts = starts[b:b + batch_size]
                xb = np.zeros((len(batch_starts), C, SEQ_LEN), np.float32)
                mb = np.zeros((len(batch_starts), SEQ_LEN), np.float32)
                ob = np.zeros((len(batch_starts), C, SEQ_LEN), bool)
                for k, s in enumerate(batch_starts):
                    seg = slice(s, s + min(SEQ_LEN, T))
                    xb[k, :, pad:] = x[seg].T
                    ob[k, :, pad:] = obs[seg].T
                    # a timestep is usable when every channel was observed
                    mb[k, pad:] = obs[seg].all(axis=1)
                out = self.model.reconstruct(x_enc=torch.from_numpy(xb).to(self.device),
                                             input_mask=torch.from_numpy(mb).to(self.device))
                rec = out.reconstruction.detach().float().cpu().numpy()
                se = (rec - xb) ** 2                                   # B x C x L
                valid = ob & (mb[:, None, :] > 0)
                for k, s in enumerate(batch_starts):
                    n = min(SEQ_LEN, T)
                    e = se[k, :, pad:pad + n].T
                    v = valid[k, :, pad:pad + n].T
                    err_sum[s:s + n] += np.where(v, e, 0.0)
                    err_cnt[s:s + n] += v
        elapsed = time.perf_counter() - t0
        with np.errstate(invalid="ignore", divide="ignore"):
            per_ch = err_sum / err_cnt
        per_ch[err_cnt == 0] = np.nan
        all_nan = np.all(np.isnan(per_ch), axis=1)
        score = np.full(T, np.nan)
        if C and (~all_nan).any():
            score[~all_nan] = np.nanmean(per_ch[~all_nan], axis=1)
        contrib = pd.DataFrame(per_ch, columns=self.signals, index=df_1hz.index)
        return score, contrib, {"seconds": elapsed, "windows": len(starts), "device": self.device}


def model_files_present(model_dir: Path = MODEL_DIR) -> bool:
    return (model_dir / "config.json").exists()
