"""Choose the 'learn your car' adaptation settings on the CALIBRATION partition.

Never uses test trips. Compares, on calibration trips only:
  base model (no adaptation) / ungated adaptation / evidence-gated adaptation (k=2, k=3)
for the same car and for the simulated other car (see evaluate_personal.py).
The setting used by the app (ai/personal.py: GATE_K=2, SCALE_CLIP=(0.8, 1.2), mapped
typical ranges) was chosen from this output; evaluate_personal.py then measures it
once on the test trips.

Usage:  python experiments/select_personal_settings.py
Output: data/processed/personal_selection.json
"""
from __future__ import annotations

import json

import numpy as np

from common import PROCESSED, ROOT, load_partition, toyota_split
from evaluate_personal import judge, other_car

import ai.personal as P
from ai.normal_behavior import NormalBehaviorModel

VARIANTS = [("base model (no adaptation)", None, None, None),
            ("ungated, scale clip 0.8-1.2", (0.8, 1.2), 3 * 3600, 0.0),
            ("gated k=2 (chosen)", (0.8, 1.2), 3 * 3600, 2.0),
            ("gated k=3", (0.8, 1.2), 3 * 3600, 3.0)]


def main() -> None:
    cal = load_partition(toyota_split()["calibration"])
    base = NormalBehaviorModel.load(ROOT / "models" / "normal_behavior")
    rng = np.random.default_rng(5)
    names = sorted(cal)
    draws = [list(rng.choice(names, 3, replace=False)) for _ in range(3)]
    draws = [d if sum(len(cal[n]) for n in d) >= 1800 else d + [n for n in names if n not in d][:2] for d in draws]
    out = []
    for label, clip, own, gate in VARIANTS:
        if clip:
            P.SCALE_CLIP, P.OWN_RANGE_MIN_S, P.GATE_K = clip, own, gate
        for part, trips in (("same_car", cal), ("simulated_other_car", {n: other_car(f) for n, f in cal.items()})):
            fa, oh = [], []
            for d in draws:
                tr, cl = P.split_reference(list(range(len(d))))
                m = base if clip is None else P.adapt_base_model(
                    base, [trips[d[i]] for i in tr], [trips[d[i]] for i in cl], {"version": "selection"})
                j = judge(m, {n: f for n, f in trips.items() if n not in d}, np.random.default_rng(99))
                fa.append(j["false_alerts_per_hour"])
                c = j["strict_detection"]["coolant_overheat"]
                oh.append(c[0] / max(1, c[1]))
            row = {"variant": label, "part": part, "false_alerts_per_hour": round(float(np.mean(fa)), 2),
                   "overheat_detection_model_only": round(float(np.mean(oh)), 2)}
            out.append(row)
            print(row, flush=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "personal_selection.json").write_text(json.dumps({"partition": "calibration", "draws": draws,
                                                                   "results": out}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
