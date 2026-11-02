"""Scenario lab and 'what if?' analysis.

Scenario lab: inject a chosen artificial fault into a COPY of a trip, analyse
original and modified copies with the same detectors, and report per detector
whether it caught the fault (strict: an alert on the injected signal that was
not already there) and how fast, plus which findings appeared or disappeared.

What if?: replace one signal over a time range by what the expected-vs-actual
model says it should have been ("as if it had stayed normal"), re-analyse, and
report which findings disappear and which persist. This is an experiment on
the MODEL - it does not prove that repairing a part would remove a problem.
Nothing here is saved to history; modified copies are labelled synthetic.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from ai.events import AnalysisResult
from ai.explain import FAMILY, Finding, findings
from ai.model_registry import ModelRegistry
from ai.pipeline import run_analysis
from ai.rules_baseline import RuleConfig
from core.analytics.injection import InjectedEvent, inject
from core.preprocessing.resample import resample_1hz
from core.schema import TIME_COL, DataOrigin, NormalizedSession

# kind -> (Arabic label, English label, allowed signals or None = any numeric, takes magnitude)
SCENARIO_KINDS = {
    "coolant_overheat": ("الحرارة بتعلى لحد 112°م", "Coolant rises to 112 °C", ["coolant_temp_c"], False),
    "voltage_drop": ("الفولت بيقع فجأة لـ 11.4", "Voltage suddenly drops to 11.4 V", ["battery_voltage"], False),
    "voltage_decline": ("الفولت بيقل بالتدريج", "Voltage declines gradually", ["battery_voltage"], True),
    "flatline": ("الحساس بيثبت على قيمة واحدة", "Sensor freezes at one value", None, False),
    "spike_noise": ("تشويش في القراءة", "Noisy reading", None, False),
    "dropout": ("الحساس بيقطع (مفيش قراءات)", "Sensor drops out (no readings)", None, False),
    "step_jump": ("الحساس بيقفز بقيمة ثابتة", "Sensor jumps by a fixed offset", None, True),
}
DETECTORS = ("rules", "consistency", "expected_vs_actual", "isolation_forest", "moment")


def _det_of(detector: str) -> str:
    if detector.startswith("rule:"):
        return "rules"
    if detector.startswith("consistency:"):
        return "consistency"
    return detector.split(":", 1)[-1]


def _same(a: Finding, b: Finding) -> bool:
    fam = FAMILY.get(a.signal, a.signal) == FAMILY.get(b.signal, b.signal)
    return fam and a.t_start_s <= b.t_end_s + 10 and b.t_start_s <= a.t_end_s + 10


def compare_findings(before: list[Finding], after: list[Finding]) -> dict:
    return {"new": [f for f in after if not any(_same(f, g) for g in before)],
            "gone": [f for f in before if not any(_same(f, g) for g in after)],
            "kept": [f for f in after if any(_same(f, g) for g in before)]}


@dataclass
class ScenarioResult:
    original: NormalizedSession
    modified: NormalizedSession
    events: list[InjectedEvent]
    result_before: AnalysisResult
    result_after: AnalysisResult
    findings_before: list[Finding]
    findings_after: list[Finding]
    per_detector: dict = field(default_factory=dict)   # det -> {ran, caught, delay_s}
    diff: dict = field(default_factory=dict)


def _grid(session: NormalizedSession) -> pd.DataFrame:
    df, _ = resample_1hz(session.data, session.signals)
    return df


def _copy_session(session: NormalizedSession, df: pd.DataFrame, events: list[InjectedEvent],
                  note: str) -> NormalizedSession:
    meta = copy.deepcopy(session.meta)
    meta.origin = DataOrigin.SYNTHETIC
    meta.source_name = f"lab:{session.meta.source_name}"
    meta.injected_events = [*meta.injected_events, *[e.to_dict() for e in events]]
    meta.notes = [*meta.notes, note]
    return NormalizedSession(df, meta)


def run_scenario(session: NormalizedSession, events: list[InjectedEvent], registry: ModelRegistry,
                 rule_cfg: RuleConfig | None = None, before: AnalysisResult | None = None) -> ScenarioResult:
    base = _grid(session)
    orig = NormalizedSession(base, session.meta)
    mod = _copy_session(session, inject(base, events), events,
                        "SCENARIO LAB: artificial faults injected into a copy of the trip.")
    # A stored result may have different rules, model weights or vehicle assignment.
    r0 = run_analysis(orig, registry, rule_cfg)
    r1 = run_analysis(mod, registry, rule_cfg)
    f0, f1 = findings(r0, orig), findings(r1, mod)
    per = {}
    for det in DETECTORS:
        st = r1.detector_status.get(det, {})
        ran = st.get("status") == "ok"
        caught, delays = [], []
        for ev in events:
            def hit(a):
                return (_det_of(a.detector) == det and ev.signal in a.signals
                        and a.t_end_s >= ev.start_s and a.t_start_s < ev.end_s)
            pre = any(hit(a) for a in r0.alerts)
            hits = [a for a in r1.alerts if hit(a)]
            ok = bool(hits) and not pre
            caught.append(ok)
            if ok:
                delays.append(max(0.0, min(a.t_start_s for a in hits) - ev.start_s))
        per[det] = {"ran": ran, "caught": caught, "delay_s": min(delays) if delays else None,
                    "reason": st.get("reason")}
    return ScenarioResult(orig, mod, events, r0, r1, f0, f1, per, compare_findings(f0, f1))


def what_if_normal(session: NormalizedSession, signal: str, t0: float, t1: float, registry: ModelRegistry,
                   rule_cfg: RuleConfig | None = None, before: AnalysisResult | None = None) -> ScenarioResult:
    """'What if `signal` had stayed normal between t0 and t1?' - replaced by the model's expected value."""
    nbm_model = registry.expected_vs_actual(session.signals)
    if nbm_model is None or signal not in nbm_model.models:
        raise ValueError(f"No expected-value model for '{signal}', so a 'normal' version cannot be built.")
    base = _grid(session)
    pred = nbm_model.predict(base)
    col = f"{signal}__expected"
    if col not in pred:
        raise ValueError(f"'{signal}' is not available in this trip.")
    cf = base.copy()
    m = (cf[TIME_COL] >= t0) & (cf[TIME_COL] <= t1)
    cf.loc[m, signal] = pred.loc[m, col]
    orig = NormalizedSession(base, session.meta)
    mod = _copy_session(session, cf, [], f"WHAT-IF: {signal} replaced by its expected value between "
                                         f"{t0:.0f} and {t1:.0f} s (experiment on the model, not a repair).")
    r0 = run_analysis(orig, registry, rule_cfg)
    r1 = run_analysis(mod, registry, rule_cfg)
    f0, f1 = findings(r0, orig), findings(r1, mod)
    return ScenarioResult(orig, mod, [], r0, r1, f0, f1, {}, compare_findings(f0, f1))
