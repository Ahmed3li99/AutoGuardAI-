"""Read-only OBD scheduler using expected information gain per estimated query cost.

Hypotheses describe observation patterns, not mechanical diagnoses. Likelihoods
are explicit heuristics; displayed belief weights are not calibrated fault risks.
"""
from __future__ import annotations

import math

import numpy as np

PATTERNS = ["normal", "voltage_pattern", "thermal_pattern", "rpm_speed_pattern", "missing_readings"]
BOUNDS = {"rpm":(500,4500),"speed_kmh":(2,130),"coolant_temp_c":(70,105),
          "battery_voltage":(12.6,14.8),"engine_load_pct":(5,90),"throttle_pct":(2,90),
          "intake_air_temp_c":(-10,70),"manifold_pressure_kpa":(20,100)}


def entropy(p):
    return float(-np.sum(p*np.log2(np.maximum(p,1e-12))))


class ActiveScheduler:
    def __init__(self, max_queries=4, budget_s=1., max_age_s=5.):
        if max_queries<1 or budget_s<=0 or max_age_s<=0:
            raise ValueError("Query count, budget and maximum age must be positive.")
        self.max_queries=max_queries; self.budget_s=budget_s; self.max_age_s=max_age_s
        self.belief=np.full(len(PATTERNS),1/len(PATTERNS))
        self.last={}; self.cost={}; self.trace=[]

    @staticmethod
    def likelihood(signal):
        # outcomes: below expected band, inside, above, no response
        table=np.tile([.08,.82,.08,.02],(len(PATTERNS),1))
        table[4]=[.05,.20,.05,.70]
        if signal=="battery_voltage": table[1]=[.65,.20,.13,.02]
        if signal in ("coolant_temp_c","intake_air_temp_c"): table[2]=[.08,.20,.70,.02]
        if signal in ("rpm","speed_kmh","engine_load_pct"): table[3]=[.35,.28,.35,.02]
        return table

    def information_gain(self,signal):
        likelihood=self.likelihood(signal)
        outcomes=self.belief@likelihood
        expected=sum(float(outcomes[j])*entropy(self.belief*likelihood[:,j]/outcomes[j]) for j in range(4))
        return max(0.,entropy(self.belief)-expected)

    def plan(self, supported, now):
        if not math.isfinite(now): raise ValueError("Invalid acquisition time.")
        candidates=[]
        for signal in sorted(set(supported)):
            age=max(0.,now-self.last[signal]) if signal in self.last else self.max_age_s*2
            gain=self.information_gain(signal); cost=max(.01,self.cost.get(signal,.1))
            overdue=age>=self.max_age_s
            priority=(1000+age if overdue else 0)+gain/cost+age*.01
            candidates.append({"signal":signal,"gain_bits":gain,"estimated_seconds":cost,
                               "reason":"refresh" if overdue else "information_gain", "priority":priority})
        candidates.sort(key=lambda d:(-d["priority"],d["signal"]))
        selected=[]; spent=0.
        for item in candidates:
            if len(selected)>=self.max_queries: break
            if spent+item["estimated_seconds"]>self.budget_s and selected: continue
            selected.append(item); spent+=item["estimated_seconds"]
        self.trace=selected
        return selected

    def observe(self, signal, value, now, elapsed=.1):
        if value is None or not np.isfinite(value): outcome=3
        else:
            lo,hi=BOUNDS.get(signal,(-np.inf,np.inf))
            outcome=0 if value<lo else 2 if value>hi else 1
        # Decay prevents certainty accumulating indefinitely from correlated samples.
        prior=.85*self.belief+.15/len(PATTERNS)
        posterior=prior*self.likelihood(signal)[:,outcome]
        self.belief=posterior/posterior.sum()
        self.last[signal]=now
        self.cost[signal]=.8*self.cost.get(signal,.1)+.2*max(.001,float(elapsed))

    def summary(self):
        return {"patterns":dict(zip(PATTERNS,self.belief.tolist())),"next_queries":self.trace,
                "interpretation":"heuristic belief weights, not fault probabilities"}


def simulate_acquisition(frame, supported=None, max_queries=4):
    scheduler=ActiveScheduler(max_queries=max_queries)
    supported=list(supported or [c for c in frame.columns if c!="t_s"])
    trace=[]
    for _,row in frame.iterrows():
        now=float(row.t_s)
        plan=scheduler.plan(supported,now)
        for item in plan:
            scheduler.observe(item["signal"],row.get(item["signal"]),now)
        trace.append({"t_s":now,"queries":plan,"patterns":scheduler.summary()["patterns"]})
    return {"trace":trace,"query_count":sum(len(t["queries"]) for t in trace),
            "fixed_scan_count":len(frame)*len(supported),"final":scheduler.summary(),
            "mode":"recorded replay simulation; not a hardware benchmark"}
