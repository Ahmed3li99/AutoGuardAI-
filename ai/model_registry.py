"""Discovers which detectors are usable on this machine and why not."""
from __future__ import annotations

import logging
from pathlib import Path

from ai.isolation_forest_baseline import IFModel
from ai.moment_inference import MODEL_DIR as MOMENT_DIR
from ai.moment_inference import MomentScorer, MomentUnavailable
from ai.normal_behavior import NormalBehaviorModel
from core.paths import MODELS_DIR

log = logging.getLogger(__name__)
IFOREST_DIR = MODELS_DIR / "iforest"
NBM_DIR = MODELS_DIR / "normal_behavior"
NBM_DRIVETRAIN_DIR = MODELS_DIR / "normal_behavior_drivetrain"
FULL_INPUTS = ["rpm", "speed_kmh", "coolant_temp_c", "engine_load_pct", "throttle_pct",
               "manifold_pressure_kpa", "intake_air_temp_c", "battery_voltage"]
FULL_MIN = 6      # at least 6 of the 8 inputs present -> full model


def choose_variant(signals: list[str] | None) -> str:
    if signals is None:
        return "full"
    present = sum(1 for c in FULL_INPUTS if c in signals)
    if present >= FULL_MIN or not {"rpm", "speed_kmh"} <= set(signals):
        return "full"
    return "drivetrain"


class ModelRegistry:
    def __init__(self, iforest_dir: Path = IFOREST_DIR, moment_dir: Path = MOMENT_DIR,
                 moment_device: str = "auto", enable_moment: bool = True, nbm_dir: Path = NBM_DIR,
                 vehicle_id: int | None = None, nbm_drivetrain_dir: Path = NBM_DRIVETRAIN_DIR,
                 powertrain: str = ""):
        self.iforest_dir, self.moment_dir, self.nbm_dir = iforest_dir, moment_dir, nbm_dir
        self.nbm_drivetrain_dir = nbm_drivetrain_dir
        self.vehicle_id = vehicle_id     # if this car has a personal model, it replaces the base one
        self.powertrain = powertrain     # "", "ice", "hybrid" or "ev" from the car profile
        self.moment_device, self.enable_moment = moment_device, enable_moment
        self._iforest: IFModel | None = None
        self._moment: MomentScorer | None = None
        self._nbm_cache: dict[str, NormalBehaviorModel | None] = {}
        self._nbm_status: dict[str, dict] = {}
        self.status: dict[str, dict] = {
            "rules": {"status": "ok", "version": "rules-v1", "kind": "heuristic_rule"}}

    def expected_vs_actual(self, signals: list[str] | None = None) -> NormalBehaviorModel | None:
        """The expected-vs-actual model that matches the available sensors:
        'full' (8 inputs) when >= FULL_MIN of them are present, otherwise the
        'drivetrain' variant (RPM + speed only). A personal model for this car is
        used when it was learned for the same variant."""
        variant = choose_variant(signals)
        if variant in self._nbm_cache:
            self.status["expected_vs_actual"] = self._nbm_status[variant]
            return self._nbm_cache[variant]
        model, st = None, {"kind": "ml_anomaly", "variant": variant}
        try:
            from ai.personal import has_personal_model, load_personal_model
            if has_personal_model(self.vehicle_id):
                pm = load_personal_model(self.vehicle_id)
                if pm.meta.get("variant", "full") == variant:
                    model, st["scope"] = pm, "vehicle"
            if model is None:
                model = NormalBehaviorModel.load(self.nbm_dir if variant == "full" else self.nbm_drivetrain_dir)
                st["scope"] = "base"
            st.update(status="ok", version=model.version, trained_on=model.meta.get("trained_on"),
                      warning=model.meta.get("load_warning"))
            if variant == "drivetrain":
                st["note_model"] = ("Only RPM/speed-type sensors available: using the drivetrain model "
                                    "(RPM and speed only).")
        except FileNotFoundError:
            st.update(status="unavailable", reason="Model file not found (run training script).")
        except Exception as e:
            log.exception("expected-vs-actual load failed")
            st.update(status="unavailable", reason=f"{type(e).__name__}: {e}")
        self._nbm_cache[variant], self._nbm_status[variant] = model, st
        self.status["expected_vs_actual"] = st
        return model

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
        self.expected_vs_actual()
        self.iforest()
        self.moment()
        return self.status
