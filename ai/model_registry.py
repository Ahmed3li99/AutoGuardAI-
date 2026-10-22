"""Discovers which detectors are usable on this machine and why not."""
from __future__ import annotations

import logging
from pathlib import Path

from ai.isolation_forest_baseline import IFModel
from ai.moment_inference import MODEL_DIR as MOMENT_DIR
from ai.moment_inference import MomentScorer, MomentUnavailable
from core.paths import MODELS_DIR

log = logging.getLogger(__name__)
IFOREST_DIR = MODELS_DIR / "iforest"


class ModelRegistry:
    def __init__(self, iforest_dir: Path = IFOREST_DIR, moment_dir: Path = MOMENT_DIR,
                 moment_device: str = "auto", enable_moment: bool = True):
        self.iforest_dir, self.moment_dir = iforest_dir, moment_dir
        self.moment_device, self.enable_moment = moment_device, enable_moment
        self._iforest: IFModel | None = None
        self._moment: MomentScorer | None = None
        self.status: dict[str, dict] = {
            "rules": {"status": "ok", "version": "rules-v1", "kind": "heuristic_rule"}}

    def iforest(self) -> IFModel | None:
        if self._iforest is None and "isolation_forest" not in self.status:
            try:
                self._iforest = IFModel.load(self.iforest_dir)
                self.status["isolation_forest"] = {
                    "status": "ok", "version": self._iforest.version, "kind": "ml_anomaly",
                    "trained_on": self._iforest.meta.get("trained_on"),
                    "warning": self._iforest.meta.get("load_warning")}
            except FileNotFoundError:
                self.status["isolation_forest"] = {"status": "unavailable", "kind": "ml_anomaly",
                                                   "reason": "Model file not found (run calibration script)."}
            except Exception as e:
                log.exception("Isolation Forest load failed")
                self.status["isolation_forest"] = {"status": "unavailable", "kind": "ml_anomaly",
                                                   "reason": f"{type(e).__name__}: {e}"}
        return self._iforest

    def moment(self) -> MomentScorer | None:
        if not self.enable_moment:
            self.status["moment"] = {"status": "unavailable", "kind": "ml_anomaly",
                                     "reason": "Disabled in settings."}
            return None
        if self._moment is None and "moment" not in self.status:
            try:
                self._moment = MomentScorer.load(self.moment_dir, self.moment_device)
                self.status["moment"] = {"status": "ok", "version": self._moment.version, "kind": "ml_anomaly",
                                         "device": self._moment.device,
                                         "trained_on": "pretrained (AutonLab); threshold calibrated on "
                                                       + str(self._moment.meta.get("calibrated_on"))}
            except MomentUnavailable as e:
                self.status["moment"] = {"status": "unavailable", "kind": "ml_anomaly", "reason": str(e)}
        return self._moment

    def load_all(self) -> dict[str, dict]:
        self.iforest()
        self.moment()
        return self.status
