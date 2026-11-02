"""File-based federated GNN training: clients keep trips; coordinator sees deltas.

Trusted participants only. Bounded updates and strict numeric formats reduce
accidents/poisoning; this is NOT secure aggregation or differential privacy.
Manual file transport permits separate PCs without an always-on cloud server.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import uuid

import numpy as np

from ai.telemetry_learning import load_npz, save_npz


@dataclass
class ClientUpdate:
    client_id: str
    round_id: str
    base_hash: str
    samples: int
    delta: dict

    def save(self,path):
        save_npz(path,{"format":"autoguard-update-v1","client_id":self.client_id,"round_id":self.round_id,
                       "base_hash":self.base_hash,"samples":self.samples}, self.delta)

    @classmethod
    def load(cls,path):
        m,a=load_npz(path)
        if m.get("format")!="autoguard-update-v1":
            raise ValueError("Not a federated client update.")
        return cls(m["client_id"],m["round_id"],m["base_hash"],m["samples"],a)


def local_update(base, frames, client_id, round_id, epochs=5):
    if not client_id or not round_id:
        raise ValueError("Client and round identifiers are required.")
    local=copy.deepcopy(base)
    count=local.train(frames,epochs=epochs)
    return ClientUpdate(str(client_id),str(round_id),base.fingerprint(),count,
                        {k:local.params[k]-base.params[k] for k in base.params})


def validate_update(base, update, round_id):
    if not isinstance(update.client_id,str) or not update.client_id:
        raise ValueError("Missing client identity.")
    if not round_id or base.meta.get("round_id")!=round_id or update.round_id!=round_id or update.base_hash!=base.fingerprint():
        raise ValueError("Update belongs to another round or another base model.")
    if type(update.samples) is not int or not 1<=update.samples<=1000000 or set(update.delta)!=set(base.params):
        raise ValueError("Invalid update schema or sample count.")
    for k,a in update.delta.items():
        if a.shape!=base.params[k].shape or not np.isfinite(a).all() or np.max(np.abs(a))>1e6:
            raise ValueError("Invalid update parameter dimensions or values.")


def aggregate(base, updates, round_id, clip_norm=2.):
    if not isinstance(round_id, str) or not round_id or base.meta.get("round_id") != round_id:
        raise ValueError("Start a round and use its designated base model.")
    if len(updates)<2 or not np.isfinite(clip_norm) or clip_norm<=0:
        raise ValueError("A round needs at least two clients and a positive update bound.")
    seen=set(); weighted={k:np.zeros_like(v) for k,v in base.params.items()}; total=0
    audit=[]
    for update in updates:
        validate_update(base,update,round_id)
        if not isinstance(update.client_id,str) or not update.client_id or update.client_id in seen:
            raise ValueError("Duplicate or missing client: refusing replay/double weighting.")
        norm=float(np.sqrt(sum(float(np.sum(a*a)) for a in update.delta.values())))
        factor=min(1.,clip_norm/max(norm,1e-12)); weight=min(update.samples,10000)
        for k,a in update.delta.items():
            weighted[k]+=weight*factor*a
        total+=weight; seen.add(update.client_id)
        audit.append({"client":update.client_id,"weight":weight,"update_norm":norm,"clipped":factor<1})
    model=copy.deepcopy(base)
    for k in model.params:
        model.params[k]+=weighted[k]/total
    # Thresholds cannot be reused after changing weights. Recalibration is mandatory.
    model.meta.update(calibrated=False,federated_round=round_id,clients=len(seen),update_audit=audit)
    return model


def new_round(base):
    model=copy.deepcopy(base)
    model.meta["round_id"]=uuid.uuid4().hex
    return model
