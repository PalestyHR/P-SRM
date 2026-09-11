
"""Load trained P-SRM weights and recover unchanged rejected candidates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .model import PointQualitySidecar, PointQualitySidecarConfig

def _affine(values, parameters):
    values=np.asarray(values,dtype=np.float64)
    mean=np.asarray(parameters["mean"],dtype=np.float64)
    scale=np.asarray(parameters["scale"],dtype=np.float64)
    coefficient=np.asarray(parameters["coefficient"],dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=len(mean):
        raise ValueError("Readout feature dimensions do not match its fitted parameters")
    return ((values-mean)/scale)@coefficient+float(parameters["intercept"])

def _sigmoid(values):
    values=np.asarray(values,dtype=np.float64)
    result=np.empty_like(values)
    positive=values>=0
    result[positive]=1/(1+np.exp(-values[positive]))
    exp=np.exp(values[~positive])
    result[~positive]=exp/(1+exp)
    return result

def recover_candidates(positions, native_valid, rejected_scores, threshold):
    """Return copied positions and final validity; never alter a coordinate.

    Scores follow the order of positions[~native_valid]. Original native
    acceptances remain accepted. Calibrate the threshold on source data.
    """
    positions=np.asarray(positions)
    native_valid=np.asarray(native_valid,dtype=bool)
    scores=np.asarray(rejected_scores,dtype=np.float64)
    if positions.ndim!=2 or native_valid.shape!=(len(positions),):
        raise ValueError("Positions and native validity must share a row axis")
    rejected=np.flatnonzero(~native_valid)
    if scores.shape!=(len(rejected),) or not np.isfinite(scores).all():
        raise ValueError("One finite score is required for each rejected candidate")
    final_valid=native_valid.copy()
    final_valid[rejected]=scores>=float(threshold)
    return positions.copy(),final_valid

class RecoveryModel:
    """Frozen quality network plus two fitted fusion stages.

    A model directory contains quality.pt and readout.json.
    """
    def __init__(self, model_dir, device="cpu"):
        directory=Path(model_dir)
        self.spec=json.loads((directory/"readout.json").read_text(encoding="utf-8"))
        payload=torch.load(directory/"quality.pt",map_location="cpu",weights_only=True)
        self.device=torch.device(device)
        self.network=PointQualitySidecar(PointQualitySidecarConfig(**payload["config"])).to(self.device).eval()
        self.network.load_state_dict(payload["model"],strict=True)
        self.threshold=float(self.spec["threshold"])
    def score(self,dense_evidence,candidate_metadata,history,native_margin):
        """Score rejected candidates using Q, the 21-D causal history, and M."""
        dense=np.asarray(dense_evidence,dtype=np.float32)
        metadata=np.asarray(candidate_metadata,dtype=np.float32)
        history=np.asarray(history,dtype=np.float32)
        margin=np.asarray(native_margin,dtype=np.float64)
        n=len(dense)
        if dense.shape!=(n,3,32,32) or metadata.shape!=(n,4) or history.shape!=(n,21) or margin.shape!=(n,):
            raise ValueError("Expected dense [N,3,32,32], metadata [N,4], history [N,21], and margin [N]")
        if not all(np.isfinite(x).all() for x in (dense,metadata,history,margin)):
            raise ValueError("P-SRM inputs must be finite")
        if n==0:return np.empty(0,dtype=np.float64)
        with torch.inference_mode(),torch.autocast(device_type=self.device.type,enabled=False):
            q=self.network(torch.as_tensor(dense,device=self.device),torch.as_tensor(metadata,device=self.device)).cpu().numpy()
        hidden=_affine(np.column_stack([q,history]),self.spec["history"])
        if self.spec["history_logit_dtype"]=="float32":hidden=hidden.astype(np.float32)
        elif self.spec["history_logit_dtype"]!="float64":raise ValueError("Unknown history logit dtype")
        scores=_sigmoid(_affine(np.column_stack([hidden,margin]),self.spec["margin"]))
        return scores.astype(self.spec["score_dtype"])
    def recover(self,positions,native_valid,dense_evidence,candidate_metadata,history,native_margin):
        """Evidence arrays contain rejected candidates only."""
        scores=self.score(dense_evidence,candidate_metadata,history,native_margin)
        return recover_candidates(positions,native_valid,scores,self.threshold)
