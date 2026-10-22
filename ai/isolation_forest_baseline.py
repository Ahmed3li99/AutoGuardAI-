"""Isolation Forest anomaly baseline on per-second window features.

Fitted only on the TRAIN partition (whole trips). The scaler is part of the
pipeline and is therefore also fitted only on train data."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai.thresholds import Threshold
from core.data_sources.files import sha256_file
from core.preprocessing.features import window_features

WINDOW_S = 30
SEED = 1234


@dataclass
class IFModel:
    pipeline: Pipeline
    signals: list[str]
    feature_names: list[str]
    meta: dict
    threshold: Threshold | None = None

    @property
    def version(self) -> str:
        return self.meta.get("version", "iforest-unknown")

    def missing_signals(self, available: list[str]) -> list[str]:
        return [s for s in self.signals if s not in available]

    def score(self, df_1hz: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Return (scores, per-signal contribution proxy). Higher = more unusual.
        Rows with incomplete features get NaN (shown as insufficient data)."""
        miss = self.missing_signals(list(df_1hz.columns))
        if miss:
            raise ValueError(f"Isolation Forest needs signals not present: {miss}")
        X = window_features(df_1hz, self.signals, WINDOW_S)[self.feature_names]
        ok = X.notna().all(axis=1).to_numpy()
        scores = np.full(len(X), np.nan)
        if ok.any():
            scores[ok] = -self.pipeline.score_samples(X[ok].to_numpy())
        # Contribution proxy: |z| of each raw signal under the train scaler.
        scaler: StandardScaler = self.pipeline.named_steps["scaler"]
        z = (X.to_numpy() - scaler.mean_) / scaler.scale_
        zdf = pd.DataFrame(np.abs(z), columns=self.feature_names, index=X.index)
        contrib = pd.DataFrame({s: zdf[[f for f in self.feature_names if f == s or f.startswith(s + "__")]]
                                .mean(axis=1) for s in self.signals})
        return scores, contrib

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, directory / "iforest.joblib")
        meta = {**self.meta, "signals": self.signals, "feature_names": self.feature_names,
                "threshold": self.threshold.to_dict() if self.threshold else None,
                "artifact_sha256": sha256_file(directory / "iforest.joblib")}
        (directory / "iforest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "IFModel":
        # joblib unpickles code, so only the artifact produced by
        # experiments/calibrate_thresholds.py and shipped in models/ is loaded,
        # and only if its SHA-256 matches the hash recorded at training time.
        meta = json.loads((directory / "iforest_meta.json").read_text(encoding="utf-8"))
        if sha256_file(directory / "iforest.joblib") != meta.get("artifact_sha256"):
            raise ValueError("iforest.joblib hash does not match iforest_meta.json - refusing to load.")
        if meta.get("sklearn_version") != sklearn.__version__:
            meta["load_warning"] = (f"Trained with scikit-learn {meta.get('sklearn_version')}, "
                                    f"running {sklearn.__version__}")
        thr = Threshold(**meta["threshold"]) if meta.get("threshold") else None
        return cls(joblib.load(directory / "iforest.joblib"), meta["signals"], meta["feature_names"],
                   meta, thr)


def fit(train_frames: list[pd.DataFrame], signals: list[str], meta: dict) -> IFModel:
    X = pd.concat([window_features(f, signals, WINDOW_S) for f in train_frames], ignore_index=True)
    X = X.dropna()
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("iforest", IsolationForest(n_estimators=200, max_samples=4096,
                                                 random_state=SEED, n_jobs=-1))])
    pipe.fit(X.to_numpy())
    meta = {**meta, "n_train_rows": int(len(X)), "window_s": WINDOW_S, "seed": SEED,
            "sklearn_version": sklearn.__version__}
    return IFModel(pipe, signals, list(X.columns), meta)
