"""Named-fault classification - DISABLED.

No open, legally reusable fault classifier with a documented schema matching
passenger-car OBD PIDs was found, and none of the obtained datasets supports
training one defensibly:
  * DieselOBD: DTC tags are constant per vehicle (8 vehicles, 7 DTC
    signatures); every signature occurs in both shipped train and test files
    and 77% of training rows are exact duplicates, so row-level accuracy would
    measure vehicle identity, not fault detection.
  * Scania APS: anonymized heavy-truck features, unrelated to car OBD PIDs.
See docs/DATASET_AUDIT.md and experiments/evaluate_classifier_optional.py.
"""
from __future__ import annotations

UNAVAILABLE_MESSAGE = "Specific fault classification unavailable for this dataset/model"


def classifier_status() -> dict:
    return {"status": "unavailable", "kind": "fault_classifier", "reason": UNAVAILABLE_MESSAGE,
            "details": "No compatible, evaluated classifier exists for passenger-car OBD data in this project."}
