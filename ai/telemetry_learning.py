"""Causal, masked telemetry windows shared by the advanced research models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS = ["rpm", "speed_kmh", "coolant_temp_c", "engine_load_pct", "throttle_pct",
           "manifold_pressure_kpa", "intake_air_temp_c", "battery_voltage"]
HISTORY = 10


def values(frame, signals):
    return frame.reindex(columns=signals).to_numpy(dtype=float)


def check_grid(frame):
    if len(frame) > 1 and not np.allclose(np.diff(frame["t_s"].to_numpy(float)), 1.0):
        raise ValueError("Advanced models require a regular 1-second grid.")


def scaling(frames, signals):
    a = np.concatenate([values(f, signals) for f in frames])
    if any(np.isfinite(a[:, i]).sum() < 30 for i in range(len(signals))):
        raise ValueError("Each model signal needs at least 30 valid training readings.")
    a[~np.isfinite(a)] = np.nan
    return np.nanmean(a, axis=0), np.maximum(np.nanstd(a, axis=0), 0.01)


def features(frame, signals, mean, scale):
    """At row t, use only observations <= t; never fill from the future."""
    check_grid(frame)
    z = (values(frame, signals) - mean) / scale
    z[~np.isfinite(z)] = np.nan
    z = np.clip(z, -30, 30)
    rolling = pd.DataFrame(z).rolling(HISTORY, min_periods=HISTORY).mean().to_numpy()
    delta = z - np.roll(z, 1, axis=0)
    if len(delta):
        delta[0] = np.nan
    observed = np.isfinite(z).astype(float)
    return np.stack([np.nan_to_num(z), np.nan_to_num(delta), np.nan_to_num(rolling), observed], axis=-1)


def examples(frames, signals, mean, scale, horizons=(1,), limit=16000):
    xs, ys, masks = [], [], []
    total = sum(max(0, len(f)-HISTORY-max(horizons)+1) for f in frames)
    for f in frames:
        if len(f) <= HISTORY + max(horizons):
            continue
        x = features(f, signals, mean, scale)
        z = (values(f, signals) - mean) / scale
        indices = np.arange(HISTORY - 1, len(f) - max(horizons))
        quota = max(1, int(np.ceil(limit*len(indices)/max(1,total))))
        if len(indices)>quota:
            indices = indices[np.linspace(0,len(indices)-1,quota,dtype=int)]
        y = np.stack([z[indices + h] for h in horizons], axis=1)
        valid = np.isfinite(y)
        keep = valid.any(axis=(1, 2)) & (x[indices, :, 3].sum(axis=1) >= 2)
        xs.append(x[indices][keep]); ys.append(np.where(valid[keep], y[keep], 0.0)); masks.append(valid[keep])
    if not xs or sum(len(x) for x in xs) < 30:
        raise ValueError("Not enough usable history/target windows; add longer reference trips.")
    x, y, m = np.concatenate(xs), np.concatenate(ys), np.concatenate(masks)
    if len(x) > limit:
        idx = np.linspace(0, len(x) - 1, limit, dtype=int)
        x, y, m = x[idx], y[idx], m[idx]
    return x, y, m


def save_npz(path, metadata, arrays):
    """Atomic single-file format; never pickle model/peer supplied objects."""
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_npz(path):
    import zipfile
    path = Path(path)
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("Model/update file is too large.")
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)>64 or len(names)!=len(set(names)):
            raise ValueError("Model archive has duplicate or excessive entries.")
        if sum(i.file_size for i in archive.infolist()) > 64 * 1024**2:
            raise ValueError("Expanded model/update is too large.")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]),parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite metadata")))
        if not isinstance(meta, dict):
            raise ValueError("Model metadata must be an object.")
        arrays = {k: data[k].copy() for k in data.files if k != "metadata"}
    if any(a.dtype.kind not in "fiu" or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Model/update contains invalid numerical parameters.")
    return meta, arrays
