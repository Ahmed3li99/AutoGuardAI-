"""Locked cross-vehicle evaluation; injected events are NOT real labeled faults."""
import json
import numpy as np

from common import ROOT, load_partition, toyota_split
from evaluate_other_cars import ved_frames, load_1hz, segments_by_gap, APM
from ai.graph_network import GraphModel
from ai.probabilistic import ForecastModel
from ai.events import segments
from core.analytics.injection import inject, InjectedEvent


def judge(model,frames):
    hours=sum(len(f) for f in frames)/3600
    events=0; compared=0; hits={"step_jump":0,"flatline":0,"spike_noise":0}; delays={k:[] for k in hits}
    for frame in frames:
        clean=model.inspect(frame)["ratios"]
        mask=np.any(clean>1,axis=1)
        events+=len(segments(mask,frame.t_s.to_numpy(),0,5))
        if len(frame)<240: continue
        compared+=1
        t0=float(frame.t_s.iloc[len(frame)//2]); t1=t0+30
        region=(frame.t_s.to_numpy()>=t0)&(frame.t_s.to_numpy()<t1)
        for kind in hits:
            changed=inject(frame,[InjectedEvent(kind,"rpm",t0,t1,"synthetic evaluation",600 if kind=="step_jump" else None)])
            bad=model.inspect(changed)["ratios"][:,model.signals.index("rpm")]
            baseline=clean[:,model.signals.index("rpm")]
            novel=region & (bad>1) & ~(baseline>1)
            if novel.any():
                hits[kind]+=1; delays[kind].append(float(frame.t_s.to_numpy()[novel][0]-t0))
    def interval(hit,n):
        if not n: return None
        z=1.96; p=hit/n; denominator=1+z*z/n
        middle=(p+z*z/(2*n))/denominator
        radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
        return [max(0.,float(middle-radius)),min(1.,float(middle+radius))]
    return {"trips":len(frames),"hours":hours,"unmodified_alert_events_per_hour":events/hours if hours else None,
            "prediction":model.evaluate(frames),
            "synthetic_events":{k:{"hit":v,"n":compared,"recall":v/compared if compared else None,
                                      "recall_wilson95":interval(v,compared),
                                      "median_delay_s":float(np.median(delays[k])) if delays[k] else None} for k,v in hits.items()}}


def main():
    split=toyota_split()
    train=list(load_partition(split["train"]).values()); cal=list(load_partition(split["calibration"]).values())
    test=list(load_partition(split["test"]).values())
    signals=["rpm","speed_kmh"]
    model=GraphModel.initialize(train,signals); model.train(train); model.calibrate(cal)
    model.meta.update(scope="dataset",dataset="Toyota carOBD",variant="drivetrain")
    outdir=ROOT/"models/advanced/drivetrain"; model.save(outdir/"graph.npz")
    forecast=ForecastModel.fit(train,cal,signals); forecast.meta.update(scope="dataset",dataset="Toyota carOBD",variant="drivetrain")
    forecast.save(outdir/"forecast.npz")
    full=GraphModel.load(ROOT/"models/advanced/graph.npz")
    old=GraphModel.load(ROOT/"experiments/baselines/graph_v1.npz")
    out={"protocol":{"seed":7,"selection":"first 12 fixed VED test vehicle IDs; first 5 usable chronological trips each",
                     "labels":"No verified mechanical-fault labels; only injected events have ground truth.",
                     "toyota":"Previously examined test partition is a regression benchmark, not a new blind test.",
                     "tuning":"No VED/APMonitor data used to train or tune these models."},
         "toyota_previous":judge(old,test),"toyota_residual":judge(full,test),"ved":[],"apmonitor":{}}
    ids=json.loads((ROOT/"datasets/splits/ved_cars_split.json").read_text())["test"][:12]
    for vid in ids:
        frames=ved_frames(vid)[:5]
        if frames:
            item={"vehicle_id":vid,**judge(model,frames),"forecast":forecast.evaluate(frames)}
            out["ved"].append(item); print(f"Evaluated VED car {vid}: {len(frames)} trips",flush=True)
    for name,path in [("Ford",APM/"automotive.txt"),("Chrysler",APM/"auto_iowa.txt")]:
        frames=segments_by_gap(load_1hz(path))
        if frames:
            out["apmonitor"][name]={**judge(model,frames),"forecast":forecast.evaluate(frames)}
    output=ROOT/"data/processed/advanced_generalization.json"
    output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf-8")
    print(output,flush=True)


if __name__=="__main__": main()
