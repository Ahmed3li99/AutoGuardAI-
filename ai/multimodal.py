"""Synchronized WAV + accelerometer + OBD fusion with a reference-fitted model.

No microphone/device is opened implicitly. Times are aligned using explicit
user-supplied offsets. Missing modalities are not silently replaced with zeros.
The model learns expected audio/vibration features conditioned on RPM and load;
it detects deviations, not named mechanical faults.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np
import pandas as pd

FEATURES=["audio_rms","audio_centroid_hz","audio_low_band_fraction","vibration_rms"]


def audio_features(path, offset_s=0.):
    if not np.isfinite(offset_s): raise ValueError("Invalid audio time offset.")
    rows=[]
    with wave.open(str(path),"rb") as stream:
        rate=stream.getframerate(); channels=stream.getnchannels()
        if stream.getsampwidth()!=2 or stream.getcomptype()!="NONE" or not 1000<=rate<=192000 or not 1<=channels<=8:
            raise ValueError("Use an uncompressed 16-bit PCM WAV, 1–8 channels, 1–192 kHz.")
        if stream.getnframes()/rate>4*3600:
            raise ValueError("Audio recording exceeds four hours; split it before importing.")
        for second in range(stream.getnframes()//rate):
            raw=np.frombuffer(stream.readframes(rate),dtype="<i2").reshape(-1,channels).astype(float)/32768
            # Average channel powers, avoiding cancellation between out-of-phase stereo channels.
            raw-=raw.mean(axis=0)
            rms=float(np.sqrt(np.mean(raw*raw)))
            power=(np.abs(np.fft.rfft(raw*np.hanning(len(raw))[:,None],axis=0))**2).mean(axis=1)
            freq=np.fft.rfftfreq(len(raw),1/rate); total=float(power.sum())
            rows.append({"t_s":offset_s+second+.5,"audio_rms":rms,
                         "audio_centroid_hz":float(freq@power/total) if total>0 else 0.,
                         "audio_low_band_fraction":float(power[freq<250].sum()/total) if total>0 else 0.})
    if not rows: raise ValueError("Audio must contain at least one complete second.")
    return pd.DataFrame(rows)


def vibration_features(path, offset_s=0.):
    if not np.isfinite(offset_s): raise ValueError("Invalid vibration time offset.")
    if Path(path).stat().st_size > 256 * 1024**2:
        raise ValueError("Accelerometer file exceeds 256 MB; split it before importing.")
    d=pd.read_csv(path,usecols=["t_s","ax","ay","az"])
    for c in d: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.replace([np.inf,-np.inf],np.nan).dropna().sort_values("t_s")
    if len(d)<20 or d.t_s.duplicated().any():
        raise ValueError("Accelerometer CSV needs unique seconds in t_s and ax/ay/az in m/s² (at least 20 samples).")
    d["second"]=np.floor(d.t_s+offset_s).astype(int)
    rows=[]
    for second,group in d.groupby("second"):
        if len(group)<10: continue
        a=group[["ax","ay","az"]].to_numpy(float, copy=True)
        a-=a.mean(axis=0)  # remove each window's DC/gravity component
        rows.append({"t_s":float(second)+.5,"vibration_rms":float(np.sqrt(np.mean(np.sum(a*a,axis=1))))})
    if not rows: raise ValueError("Need at least 10 accelerometer samples in a one-second window.")
    return pd.DataFrame(rows)


def align(telemetry,audio,vibration,tolerance=.6):
    if not .0<tolerance<=1.: raise ValueError("Alignment tolerance must be in (0, 1] seconds.")
    d=telemetry.sort_values("t_s").copy()
    if not {"rpm","engine_load_pct"}<=set(d):
        raise ValueError("Fusion requires RPM and engine_load_pct to compare like operating conditions.")
    d["t_s"]=d.t_s.astype(float)
    for stream in (audio,vibration):
        stream=stream.sort_values("t_s").copy(); stream["t_s"]=stream.t_s.astype(float)
        d=pd.merge_asof(d,stream,on="t_s",direction="nearest",tolerance=tolerance)
    return d


def context(d):
    r=d.rpm.to_numpy(float)/3000; load=d.engine_load_pct.to_numpy(float)/100
    return np.column_stack([np.ones(len(d)),r,load,r*r,r*load])


@dataclass
class FusionModel:
    weights: np.ndarray
    scales: np.ndarray
    threshold: float

    @classmethod
    def fit(cls, train, calibration):
        tr=train.dropna(subset=["rpm","engine_load_pct",*FEATURES])
        ca=calibration.dropna(subset=["rpm","engine_load_pct",*FEATURES])
        if len(tr)<30 or len(ca)<20:
            raise ValueError("Fusion needs 30 aligned training seconds and 20 separate calibration seconds.")
        x=context(tr); y=tr[FEATURES].to_numpy(float)
        weights=np.linalg.solve(x.T@x+np.diag([1e-6,.1,.1,.1,.1]),x.T@y)
        residual=y-x@weights
        scales=np.maximum(np.median(np.abs(residual-np.median(residual,axis=0)),axis=0)*1.4826,
                          np.array([1e-4,1.,.001,1e-3]))
        scores=np.max(np.abs(ca[FEATURES].to_numpy(float)-context(ca)@weights)/scales,axis=1)
        q=min(1.,np.ceil((len(scores)+1)*.99)/len(scores))
        return cls(weights,scales,max(1.,float(np.quantile(scores,q,method="higher"))))

    def score(self,aligned):
        d=aligned.copy()
        good=np.isfinite(d[["rpm","engine_load_pct",*FEATURES]].to_numpy(float)).all(axis=1)
        ratio=np.full((len(d),len(FEATURES)),np.nan)
        ratio[good]=np.abs(d.loc[good,FEATURES].to_numpy(float)-context(d.loc[good])@self.weights)/self.scales
        d["fusion_score"]=np.nan
        d.loc[good,"fusion_score"]=ratio[good].max(axis=1)/self.threshold
        d["feature"]=None
        d.loc[good,"feature"]=[FEATURES[i] for i in ratio[good].argmax(axis=1)]
        d["aligned"]=good
        return d


def analyse_files(session, audio_path, vibration_path, audio_offset=0., vibration_offset=0., reference_end=None):
    from core.preprocessing.resample import resample_1hz
    d,_=resample_1hz(session.data,session.signals)
    aligned=align(d,audio_features(audio_path,audio_offset),vibration_features(vibration_path,vibration_offset))
    complete=aligned.dropna(subset=["rpm","engine_load_pct",*FEATURES])
    if reference_end is None or not np.isfinite(reference_end):
        raise ValueError("Choose the end time of a known-normal reference interval.")
    reference=complete[complete.t_s<=reference_end]
    n=int(len(reference)*.7)
    model=FusionModel.fit(reference.iloc[:n],reference.iloc[n:])
    scored=model.score(aligned)
    scored["reference"]=scored.t_s<=reference_end
    if not ((scored.t_s>reference_end)&scored.aligned).any():
        raise ValueError("No aligned observations after the reference interval to evaluate.")
    events=scored[(scored.fusion_score>1)&~scored.reference]
    return {"timeline":scored,"events":events,"aligned_seconds":int(scored.aligned.sum()),
            "total_seconds":len(scored),"reference_seconds":len(reference),
            "threshold":model.threshold,"audio_offset_s":audio_offset,"vibration_offset_s":vibration_offset,
            "interpretation":"reference-conditioned deviations, not fault classification"}


def export_event_clip(audio_path, output_path, timeline, event_s, audio_offset=0., radius_s=5.):
    """Export the event's WAV and synchronized telemetry CSV on the trip clock."""
    output_path = Path(output_path)
    source = Path(audio_path)
    if output_path.resolve() == source.resolve():
        raise ValueError("Choose a different output file to preserve the original recording.")
    if not all(np.isfinite(v) for v in (event_s, audio_offset, radius_s)) or not 0 < radius_s <= 60:
        raise ValueError("Invalid event window.")
    with wave.open(str(source), "rb") as inp:
        rate = inp.getframerate()
        start = max(0, int((event_s-radius_s-audio_offset)*rate))
        end = min(inp.getnframes(), int((event_s+radius_s-audio_offset)*rate))
        if end <= start: raise ValueError("Event does not overlap the audio recording.")
        inp.setpos(start); raw = inp.readframes(end-start); params = inp.getparams()
    with wave.open(str(output_path), "wb") as out:
        out.setparams(params); out.writeframes(raw)
    t0, t1 = audio_offset+start/rate, audio_offset+end/rate
    rows = timeline.loc[timeline.t_s.between(t0, t1)].copy()
    rows["clip_time_s"] = rows.t_s-t0
    csv = output_path.with_suffix(".csv"); rows.to_csv(csv, index=False)
    return {"audio": str(output_path), "readings": str(csv), "trip_start_s": t0, "trip_end_s": t1}
