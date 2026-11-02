"""OPTIONAL adaptation experiment: fine-tune pretrained MOMENT-1-small with its
own pretraining objective (masked-patch reconstruction) on carOBD TRAIN trips.
The backbone is NOT trained from scratch - we continue from the published
weights for a few epochs. Model selection uses CALIBRATION trips; TEST trips
are untouched.

Reported metric: masked-patch normalized MSE on calibration windows
(1.0 = predicting the window mean), zero-shot vs adapted.

Usage:  python experiments/adapt_moment.py [--epochs 4]
Output: models/moment-1-small-carobd/ (weights + meta), data/processed/moment_adaptation.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

from common import PROCESSED, ROOT, load_partition, toyota_split

os.environ["HF_HUB_OFFLINE"] = "1"
from momentfm import MOMENTPipeline  # noqa: E402

from core.preprocessing.features import MODEL_SIGNALS  # noqa: E402

BASE = ROOT / "models" / "moment-1-small"
OUT = ROOT / "models" / "moment-1-small-carobd"
L, P, SEED = 512, 8, 11


def windows(frames, mean, std, stride):
    xs, ms = [], []
    for f in frames:
        x = ((f[MODEL_SIGNALS].to_numpy(np.float32) - mean) / std)
        obs = np.isfinite(x).all(axis=1)
        x = np.nan_to_num(x).T
        for s in range(0, max(len(f) - L, 0) + 1, stride):
            if len(f) < L:
                break
            xs.append(x[:, s:s + L])
            ms.append(obs[s:s + L].astype(np.float32))
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ms))


def masked_nmse(model, X, M, dev, k=4, bs=32):
    """Each patch hidden exactly once (k passes); error on hidden, observed steps."""
    num, den = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x, m = X[i:i + bs].to(dev), M[i:i + bs].to(dev)
            for off in range(k):
                hide = torch.ones_like(m)
                for p in range(off, L // P, k):
                    hide[:, p * P:(p + 1) * P] = 0
                rec = model.reconstruction(x_enc=x, input_mask=m, mask=hide).reconstruction
                sel = ((1 - hide) * m).unsqueeze(1).expand_as(x).bool()
                num += float(((rec - x) ** 2)[sel].sum())
                # normalise by each window-channel variance (mean predictor = 1.0)
                var = x.var(dim=2, keepdim=True).clamp_min(1e-3).expand_as(x)
                den += float(var[sel].sum())
    return num / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    split = toyota_split()
    train = list(load_partition(split["train"]).values())
    calib = list(load_partition(split["calibration"]).values())
    meta0 = json.loads((BASE / "moment_meta.json").read_text(encoding="utf-8"))
    mean, std = np.array(meta0["mean"], np.float32), np.array(meta0["std"], np.float32)
    Xtr, Mtr = windows(train, mean, std, stride=128)
    Xca, Mca = windows(calib, mean, std, stride=256)
    print("train windows", tuple(Xtr.shape), "calibration windows", tuple(Xca.shape))

    model = MOMENTPipeline.from_pretrained(str(BASE), local_files_only=True,
                                           model_kwargs={"task_name": "reconstruction",
                                                         "freeze_encoder": False, "freeze_embedder": False,
                                                         "freeze_head": False, "mask_ratio": 0.3})
    model.init()
    model.to(dev)
    res = {"device": dev, "train_windows": int(len(Xtr)), "calibration_windows": int(len(Xca)),
           "epochs": args.epochs, "lr": args.lr, "seed": SEED, "base_revision": meta0["hf_revision"]}
    res["zero_shot_masked_nmse"] = round(masked_nmse(model, Xca, Mca, dev), 4)
    print("zero-shot masked nMSE (calibration):", res["zero_shot_masked_nmse"])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    best, hist = res["zero_shot_masked_nmse"], []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            x, m = Xtr[idx].to(dev), Mtr[idx].to(dev)
            out = model.reconstruction(x_enc=x, input_mask=m)        # random 30 % patch mask
            hidden = ((1 - out.pretrain_mask) * m).unsqueeze(1).expand_as(x)
            loss = (((out.reconstruction - x) ** 2) * hidden).sum() / hidden.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        val = masked_nmse(model, Xca, Mca, dev)
        hist.append({"epoch": ep + 1, "train_loss": round(tot / len(perm), 4), "calibration_masked_nmse": round(val, 4)})
        print(hist[-1])
        if val < best:
            best = val
            OUT.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(OUT))
    res.update(history=hist, best_calibration_masked_nmse=round(best, 4), train_seconds=round(time.time() - t0, 1))
    if OUT.exists():
        shutil.copy(BASE / "moment_meta.json", OUT / "moment_meta.json")   # same scaler; threshold recalibrated later
        (OUT / "ADAPTATION.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "moment_adaptation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
