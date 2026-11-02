"""Train advanced models using disjoint recorded Toyota trip partitions."""
import json

from common import ROOT, load_partition, toyota_split
from ai.advanced_service import train_models


def main():
    split = toyota_split()
    cal_key = next(k for k in ("calibration", "calib", "val") if k in split)
    names = [split[k] for k in ("train", cal_key, "test")]
    if any(set(names[i]) & set(names[j]) for i in range(3) for j in range(i)):
        raise ValueError("Trip partitions overlap")
    frames = [list(load_partition(group).values()) for group in names]
    result = train_models(*frames, ROOT / "models" / "advanced",
                          {"scope": "dataset", "dataset": "Toyota carOBD",
                           "train_trips": names[0], "calibration_trips": names[1], "test_trips": names[2]})
    output = json.dumps(result, indent=2, allow_nan=False)
    (ROOT / "data" / "processed" / "advanced_evaluation.json").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
