"""Compute causal history from native decisions; no future-frame features."""
from pathlib import Path
import argparse
import numpy as np
from .history import CausalHistory
S3_FEATURE_NAMES = [
    "has_previous_native_valid",
    "has_two_previous_native_valid",
    "track_age_frames",
    "previous_native_valid_count",
    "previous_native_valid_fraction",
    "gap_since_last_native_valid",
    "consecutive_rejects_before_current",
    "last_valid_margin",
    "last_valid_visible_score",
    "last_valid_delta_x",
    "last_valid_delta_y",
    "last_valid_distance",
    "estimated_velocity_x",
    "estimated_velocity_y",
    "estimated_speed",
    "constant_velocity_innovation_x",
    "constant_velocity_innovation_y",
    "constant_velocity_innovation_norm",
    "historical_prediction_rms",
    "normalized_innovation",
    "last_valid_margin_trend",
]

def derive_for_query(frame,candidate,margin,visible_score,native_valid,query_frame):
    state=CausalHistory(query_frame);ids=[];features=[]
    for i,t in enumerate(frame):
        row=state.update(t,candidate[i],margin[i],visible_score[i],native_valid[i])
        if row is not None:ids.append(i);features.append(row)
    return np.asarray(ids,np.int64),np.asarray(features,np.float32).reshape(-1,21),None

def process_file(path):
    with np.load(path,allow_pickle=False) as z:
        query=z['query_index'];indices=[];features=[]
        for q in np.unique(query):
            pos=np.flatnonzero(query==q)
            chosen,h,_=derive_for_query(z['frame_index'][pos],z['candidate_xy'][pos],z['native_margin'][pos],z['native_visible_score'][pos],z['native_valid'][pos],int(z['query_frame'][pos[0]]))
            indices.extend(pos[chosen]);features.extend(h)
        order=np.argsort(indices);indices=np.asarray(indices,dtype=np.int64)[order]
        h=np.asarray(features,np.float32).reshape(-1,21)[order]
        return dict(segment_name=z['segment_name'],query_index=query[indices],frame_index=z['frame_index'][indices],s3_features=h,feature_names=np.asarray(S3_FEATURE_NAMES))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--history-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    a.output_root.mkdir(parents=True,exist_ok=True)
    for file in sorted(a.history_root.glob('*.npz')):np.savez_compressed(a.output_root/file.name,**process_file(file))
if __name__=='__main__':main()
