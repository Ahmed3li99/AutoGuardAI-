"""Small temporal graph neural network, implemented in NumPy (no GPU required).

One message-passing layer: h_i=tanh(x_i W_self + sum_j A_ij x_j W_neighbour + e_i).
New models learn a Huber-loss correction to persistence; old models remain loadable.
Trainable node embeddings and shared output weights predict the NEXT second.
The graph is estimated on training trips only. Residual thresholds are fitted on
separate calibration trips. Edges are associations, never claims of causality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd

from ai.telemetry_learning import HISTORY, examples, features, load_npz, save_npz, scaling, values


@dataclass
class GraphModel:
    signals: list[str]
    mean: np.ndarray
    scale: np.ndarray
    adjacency: np.ndarray
    params: dict[str, np.ndarray]
    threshold: np.ndarray
    meta: dict = field(default_factory=dict)

    @classmethod
    def initialize(cls, frames, signals, hidden=16, seed=7, residual=True):
        if len(signals) < 2:
            raise ValueError("A sensor graph needs at least two signals.")
        mean, scale = scaling(frames, signals)
        data = pd.concat([f.reindex(columns=signals) for f in frames], ignore_index=True)
        corr = np.nan_to_num(data.corr(min_periods=30).to_numpy())
        adj = np.abs(corr); np.fill_diagonal(adj, 0)
        for row in adj:
            order = np.argsort(row)
            row[order[:-min(3, len(signals)-1)]] = 0
        adj = (adj + adj.T) / 2
        adj /= np.maximum(adj.sum(axis=1, keepdims=True), 1e-12)
        rng = np.random.default_rng(seed)
        params = {"self": rng.normal(0, .15, (4, hidden)), "neighbour": rng.normal(0, .15, (4, hidden)),
                  "embedding": np.zeros((len(signals), hidden)), "out": rng.normal(0, .15, (hidden,)),
                  "bias": np.zeros(len(signals))}
        if residual:
            params["out"][:] = 0  # Start from persistence and learn only its correction.
        return cls(list(signals), mean, scale, adj, params, np.ones(len(signals)),
                   {"kind": "temporal-message-passing-gnn", "seed": seed, "calibrated": False,
                    "residual": bool(residual), "loss": "huber" if residual else "mse"})

    def forward(self, x):
        msg = np.einsum("ij,bjf->bif", self.adjacency, x)
        h = np.tanh(x @ self.params["self"] + msg @ self.params["neighbour"] + self.params["embedding"])
        pred = h @ self.params["out"] + self.params["bias"]
        if self.meta.get("residual", False):
            pred = pred + x[:,:,0]
        return pred, h, msg

    def train(self, frames, epochs=35, rate=.01, seed=7):
        if not 1 <= epochs <= 200:
            raise ValueError("Epochs must be between 1 and 200.")
        x, y, valid = examples(frames, self.signals, self.mean, self.scale)
        y, valid = y[:, 0], valid[:, 0]
        rng = np.random.default_rng(seed)
        mom = {k: np.zeros_like(v) for k, v in self.params.items()}
        var = {k: np.zeros_like(v) for k, v in self.params.items()}
        step = 0
        for _ in range(epochs):
            indices = rng.permutation(len(x))
            for start in range(0, len(x), 256):
                ix = indices[start:start+256]
                pred, h, msg = self.forward(x[ix])
                error = pred - y[ix]
                gradient = np.clip(error, -.02, .02) if self.meta.get("loss") == "huber" else 2*error
                d = gradient * valid[ix] / max(1, valid[ix].sum())
                dh = d[..., None] * self.params["out"] * (1-h*h)
                grads = {"out": np.einsum("bih,bi->h", h, d), "bias": d.sum(axis=0),
                         "embedding": dh.sum(axis=0), "self": np.einsum("bif,bih->fh", x[ix], dh),
                         "neighbour": np.einsum("bif,bih->fh", msg, dh)}
                step += 1
                for k, g in grads.items():
                    g = np.clip(g, -5, 5)
                    mom[k] = .9*mom[k]+.1*g; var[k] = .999*var[k]+.001*g*g
                    self.params[k] -= rate*(mom[k]/(1-.9**step))/(np.sqrt(var[k]/(1-.999**step))+1e-8)
        self.meta.update(training_windows=len(x), epochs=epochs, calibrated=False)
        return len(x)

    def calibrate(self, frames):
        x, y, mask = examples(frames, self.signals, self.mean, self.scale)
        err = np.abs(self.forward(x)[0]-y[:, 0])
        counts = mask[:, 0].sum(axis=0)
        if (counts < 30).any():
            raise ValueError("Each signal needs 30 held-out calibration targets.")
        self.threshold = np.array([max(.05, float(np.quantile(err[mask[:, 0, i], i], .99)))
                                   for i in range(len(self.signals))])
        self.meta.update(calibrated=True, calibration_windows=len(x), threshold_quantile=.99)

    def evaluate(self, frames):
        x, y, m = examples(frames, self.signals, self.mean, self.scale)
        p = self.forward(x)[0]
        return {s: {"mae": float(np.abs(p[:,i]-y[:,0,i])[m[:,0,i]].mean()*self.scale[i]),
                    "persistence_mae": float(np.abs(x[:,i,0]-y[:,0,i])[m[:,0,i]].mean()*self.scale[i]),
                    "exceedance_rate": float((np.abs(p[:,i]-y[:,0,i])[m[:,0,i]] > self.threshold[i]).mean())}
                for i,s in enumerate(self.signals) if m[:,0,i].any()}

    def inspect(self, frame):
        if not self.meta.get("calibrated"):
            raise ValueError("Calibrate the graph on separate reference trips first.")
        x = features(frame, self.signals, self.mean, self.scale)
        pred = np.full((len(frame), len(self.signals)), np.nan)
        if len(frame) > HISTORY:
            for start in range(HISTORY, len(frame), 2048):
                end = min(start+2048, len(frame))
                pred[start:end] = self.forward(x[start-1:end-1])[0]*self.scale+self.mean
        actual = values(frame, self.signals)
        ratios = np.abs(actual-pred) / (self.scale*self.threshold)
        ratios[~np.isfinite(actual)] = np.nan
        for i in range(len(frame)-1):
            if x[i,:,3].sum() < 2:
                ratios[i+1] = np.nan
        ranks = sorted([{"signal": s, "peak_ratio": float(np.nanmax(ratios[:,i])),
                         "fraction_above": float(np.mean(ratios[np.isfinite(ratios[:,i]),i]>1))}
                        for i,s in enumerate(self.signals) if np.isfinite(ratios[:,i]).any()],
                       key=lambda r:r["peak_ratio"], reverse=True)
        edges = [{"from": self.signals[j], "to": self.signals[i], "weight": float(self.adjacency[i,j])}
                 for i in range(len(self.signals)) for j in range(len(self.signals)) if self.adjacency[i,j]>0]
        return {"expected": pred, "ratios": ratios, "ranking": ranks, "edges": edges}

    def fingerprint(self):
        h = hashlib.sha256("|".join(self.signals).encode())
        h.update(str(bool(self.meta.get("residual", False))).encode())
        for a in [self.mean, self.scale, self.adjacency, *[self.params[k] for k in sorted(self.params)]]:
            h.update(np.asarray(a,dtype="<f8").tobytes())
        return h.hexdigest()

    def save(self, path):
        save_npz(path, {**self.meta, "signals": self.signals, "format": "autoguard-graph-v1"},
                 {"mean": self.mean, "scale": self.scale, "adjacency": self.adjacency,
                  "threshold": self.threshold, **{f"p_{k}":v for k,v in self.params.items()}})

    @classmethod
    def load(cls, path):
        meta, a = load_npz(path)
        if meta.get("format") != "autoguard-graph-v1":
            raise ValueError("Not an AutoGuard graph model.")
        s = meta["signals"]; n = len(s)
        if type(meta.get("residual",False)) is not bool or meta.get("loss","mse") not in ("mse","huber"):
            raise ValueError("Invalid graph prediction mode.")
        p = {k[2:]:v for k,v in a.items() if k.startswith("p_")}
        if not 2 <= n <= 32 or len(set(s)) != n or set(p) != {"self","neighbour","embedding","out","bias"}:
            raise ValueError("Invalid graph schema.")
        h = len(p["out"])
        shapes = {"self":(4,h), "neighbour":(4,h), "embedding":(n,h), "out":(h,), "bias":(n,)}
        if not 1 <= h <= 256 or any(p[k].shape!=shape for k,shape in shapes.items()):
            raise ValueError("Invalid graph parameter shapes.")
        if any(a[k].shape != (n,) for k in ("mean","scale","threshold")) or a["adjacency"].shape!=(n,n):
            raise ValueError("Invalid graph dimensions.")
        if (a["scale"]<=0).any() or (a["threshold"]<=0).any():
            raise ValueError("Invalid graph scale.")
        sums = a["adjacency"].sum(axis=1)
        if (a["adjacency"] < 0).any() or not np.all(np.isclose(sums, 0) | np.isclose(sums, 1)):
            raise ValueError("Invalid graph adjacency weights.")
        return cls(s,a["mean"],a["scale"],a["adjacency"],p,a["threshold"],meta)
