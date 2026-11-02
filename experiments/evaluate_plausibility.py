"""Measure the implausible-spike check (RPM, MAF, baro) on HELD-OUT real data only.

Limits come from experiments/derive_plausibility_limits.py (Toyota train+calibration, VED dev petrol
cars). Here: Toyota TEST trips, VED TEST petrol cars, all VED hybrids / plug-ins / EVs, Ford, Chrysler.
  - alerts per hour on every data set (no glitches are known there, except in the Ford highway file)
  - on the Ford highway file: how many of the 19 glitch events (values no healthy car produces, the
    markers defined in evaluate_other_cars.GLITCH) are now flagged, by the spike check alone and by
    all consistency checks + limit rules together

Usage:  python experiments/evaluate_plausibility.py
Output: data/processed/plausibility_evaluation.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from common import PROCESSED, load_partition, toyota_split
from evaluate_other_cars import (APM, glitch_events, hours, load_1hz, segments_by_gap, ved_frames, ved_index,
                                 ved_split)

from ai.consistency import run_consistency
from ai.rules_baseline import run_rules


def spikes(frames) -> dict:
    c = Counter(a.detector for f in frames for a in run_consistency(f) if a.detector.endswith("_spike"))
    h = hours(frames)
    return {"hours": round(h, 1), "spike_alerts": dict(c), "spike_alerts_per_hour": round(sum(c.values()) / h, 4) if h else None}


def ford_glitches(pieces) -> dict:
    n = Counter()
    hit_spike, hit_any = Counter(), Counter()
    stray = 0
    for p in pieces:
        cons, rules = run_consistency(p), run_rules(p)
        ev = glitch_events(p)
        for s, e, sig in ev:
            n[sig] += 1

            def near(a):
                return a.t_start_s <= e + 1 and a.t_end_s >= s - 1
            hit_spike[sig] += any(near(a) and a.detector.endswith("_spike") for a in cons)
            hit_any[sig] += any(near(a) for a in cons) or any(near(a) and sig in a.signals for a in rules)
        stray += sum(1 for a in cons if a.detector.endswith("_spike")
                     and not any(a.t_start_s <= e + 1 and a.t_end_s >= s - 1 for s, e, _ in ev))
    return {"glitch_events": dict(n), "total": sum(n.values()),
            "flagged_by_spike_check": dict(hit_spike), "flagged_by_spike_check_total": sum(hit_spike.values()),
            "flagged_by_all_consistency_or_rules": dict(hit_any),
            "flagged_by_all_consistency_or_rules_total": sum(hit_any.values()),
            "spike_alerts_not_on_a_glitch_event": stray}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    res = {"toyota_test": spikes(list(load_partition(toyota_split()["test"]).values()))}
    print("toyota", res["toyota_test"], flush=True)
    idx, split = ved_index(), ved_split()
    groups = {"ved_test_ice": split["test"]}
    for vtype in ("HEV", "PHEV", "EV"):
        groups[f"ved_{vtype.lower()}"] = [v for v, i in idx.items() if i.get("type") == vtype]
    for name, ids in groups.items():
        res[name] = {"cars": len(ids), **spikes([f for v in ids for f in ved_frames(v)])}
        print(name, res[name], flush=True)
    res["chrysler"] = spikes(segments_by_gap(load_1hz(APM / "auto_iowa.txt")))
    res["ford_automotive"] = spikes(segments_by_gap(load_1hz(APM / "automotive.txt")))
    trip = segments_by_gap(load_1hz(APM / "auto_trip" / "auto_trip.csv"))
    res["ford_auto_trip"] = {**spikes(trip), **ford_glitches(trip)}
    for k in ("chrysler", "ford_automotive", "ford_auto_trip"):
        print(k, res[k], flush=True)
    (PROCESSED / "plausibility_evaluation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
