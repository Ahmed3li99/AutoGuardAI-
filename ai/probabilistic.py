"""Multi-horizon regression with held-out residual trajectories and conformal bands.

Intervals are marginal 90% bands, NOT a joint 90% guarantee for an entire path.
Empirical exceedance probabilities are research estimates, not fault probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai.telemetry_learning import HISTORY, examples, features, load_npz, save_npz, scaling


def design(x):
    flat = x.reshape(len(x), -1)
    return np.column_stack([np.ones(len(x)), flat, np.tanh(flat)])


@dataclass
class ForecastModel:
    signals: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    residuals: np.ndarray
    radii: np.ndarray
    horizons: tuple = (10, 30, 60)
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, train, calibration, signals, horizons=(10,30,60), ridge=10.):
        horizons = tuple(int(h) for h in horizons)
        if len(horizons)<1 or sorted(set(horizons)) != list(horizons) or horizons[0]<1 or horizons[-1]>300:
            raise ValueError("Forecast horizons must be increasing, unique, between 1 and 300 seconds.")
        mean, scale = scaling(train, signals)
        x,y,m = examples(train, signals, mean, scale, horizons)
        d = design(x)
        weights = np.zeros((d.shape[1], len(horizons), len(signals)))
        penalty = np.eye(d.shape[1])*ridge; penalty[0,0] = 0
        for h in range(len(horizons)):
            for s in range(len(signals)):
                ok=m[:,h,s]
                if ok.sum()<30:
                    raise ValueError("Not enough training targets for every horizon and signal.")
                a=d[ok]
                weights[:,h,s]=np.linalg.solve(a.T@a+penalty, a.T@y[ok,h,s])
        xc,yc,mc = examples(calibration,signals,mean,scale,horizons,limit=4000)
        # Complete residual vectors preserve cross-signal and cross-horizon dependence.
        complete=mc.all(axis=(1,2))
        residuals=(yc-np.einsum("bf,fhs->bhs",design(xc),weights))[complete]
        if len(residuals)<30:
            raise ValueError("Need at least 30 complete calibration trajectories.")
        q=min(1.,np.ceil((len(residuals)+1)*.90)/len(residuals))
        radii=np.quantile(np.abs(residuals),q,axis=0,method="higher")
        return cls(list(signals),mean,scale,weights,residuals,radii,horizons,
                   {"kind":"ridge-joint-residual-conformal", "coverage_target":.90,
                    "training_windows":len(x),"calibration_windows":len(residuals)})

    def forecast(self, frame, paths=128, seed=7, limits=None):
        if len(frame)<HISTORY or not 1<=paths<=2000:
            raise ValueError("Need ten seconds of history and 1–2000 paths.")
        x=features(frame,self.signals,self.mean,self.scale)[-1:]
        if x[0,:,3].sum()!=len(self.signals):
            raise ValueError("All model sensors must have a current reading; choose a compatible model.")
        center=np.einsum("bf,fhs->bhs",design(x),self.weights)[0]
        rng=np.random.default_rng(seed)
        residuals=self.residuals[rng.integers(0,len(self.residuals),paths)]
        trajectories=(center+residuals)*self.scale+self.mean
        lower=(center-self.radii)*self.scale+self.mean
        upper=(center+self.radii)*self.scale+self.mean
        probabilities={}
        # Use all calibration residuals, not the random visual sample, for stable probabilities.
        population=(center+self.residuals)*self.scale+self.mean
        for s,lim in (limits or {}).items():
            if s in self.signals:
                i=self.signals.index(s)
                probabilities[s]={"limit":float(lim), "above":(population[:,:,i]>lim).mean(axis=0).tolist()}
        return {"t_s":float(frame.t_s.iloc[-1])+np.array(self.horizons), "center":center*self.scale+self.mean,
                "lower":lower,"upper":upper,"paths":trajectories,"probabilities":probabilities}

    def evaluate(self, frames):
        x,y,m=examples(frames,self.signals,self.mean,self.scale,self.horizons,limit=6000)
        pred=np.einsum("bf,fhs->bhs",design(x),self.weights)
        return {s:{str(h):{"coverage":float((np.abs(y[:,j,i]-pred[:,j,i])[m[:,j,i]]<=self.radii[j,i]).mean()),
                           "mae":float(np.abs(y[:,j,i]-pred[:,j,i])[m[:,j,i]].mean()*self.scale[i]),
                           "mean_band_width":float(2*self.radii[j,i]*self.scale[i])}
                    for j,h in enumerate(self.horizons) if m[:,j,i].any()} for i,s in enumerate(self.signals)}

    def save(self,path):
        save_npz(path,{**self.meta,"format":"autoguard-forecast-v1","signals":self.signals,"horizons":self.horizons},
                 {"mean":self.mean,"scale":self.scale,"weights":self.weights,"residuals":self.residuals,"radii":self.radii})

    @classmethod
    def load(cls,path):
        m,a=load_npz(path)
        if m.get("format")!="autoguard-forecast-v1":
            raise ValueError("Not an AutoGuard forecast model.")
        signals=m["signals"]; horizons=tuple(m["horizons"]); n=len(signals); h=len(horizons)
        if not 2<=n<=32 or len(set(signals))!=n or not 1<=h<=10 or list(horizons)!=sorted(set(horizons)) or min(horizons)<1 or max(horizons)>300:
            raise ValueError("Invalid forecast schema.")
        if (a["mean"].shape!=(n,) or a["scale"].shape!=(n,) or a["weights"].shape!=(1+8*n,h,n)
                or a["radii"].shape!=(h,n) or a["residuals"].ndim!=3
                or a["residuals"].shape[1:]!=(h,n) or len(a["residuals"])<30
                or (a["scale"]<=0).any() or (a["radii"]<0).any()):
            raise ValueError("Invalid forecast parameter dimensions.")
        return cls(signals,a["mean"],a["scale"],a["weights"],a["residuals"],a["radii"],horizons,m)
