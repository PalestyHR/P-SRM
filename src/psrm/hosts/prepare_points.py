"""Convert raw point-host exports to the common P-SRM reproduction layout."""
from pathlib import Path
import argparse,csv,shutil
import numpy as np
import pandas as pd
from ..data import build_dense_projection
from ..history_batch import process_file
from ..prepare_dataset import packet_filename,index
from .common import read_ids

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',choices=['tapnet','online-tapir','tapnext'],required=True)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for folder in ['evidence','history']:(a.output/folder).mkdir(parents=True,exist_ok=True)
    if a.host=='tapnext':
        files=(a.raw/'packets').glob('*.npz')
        for file in files:
            with np.load(file,allow_pickle=False) as z:name=str(z['segment_name'])
            shutil.copyfile(file,a.output/'evidence'/packet_filename(name))
        history=a.raw/'history';native=[a.raw/'native_per_video.csv']
    else:
        for file in sorted((a.raw/'evidence').glob('*.npz')):
            with np.load(file,allow_pickle=False) as z:
                tapnet=a.host=='tapnet';name=str(z['source_id'][0]).split('|q')[0] if tapnet else str(z['segment_name'])
                candidate=z['candidate_xy'].astype(np.float32);dense,metadata=build_dense_projection(z['task_evidence'].astype(np.float32),candidate,candidate/8.)
                np.savez_compressed(a.output/'evidence'/packet_filename(name),segment_name=np.asarray(name),query_index=z['query_index'],frame_index=z['frame' if tapnet else 'frame_index'],dense_evidence=dense,candidate_metadata=metadata,native_margin=z['native_margin'],outcome=z['outcome_cla' if tapnet else 'outcome'],candidate_error=z['candidate_error_px' if tapnet else 'candidate_error'],gt_visible=z['gt_visible'])
        history=a.raw/'native_history';native=list(a.raw.glob('per_video*.csv'))
    for file in sorted(history.glob('*.npz')):
        d=process_file(file);name=str(d['segment_name']);np.savez_compressed(a.output/'history'/packet_filename(name),**d)
        ep=a.output/'evidence'/packet_filename(name)
        if len(d['query_index'])==0 and not ep.exists():
            np.savez_compressed(ep,segment_name=np.asarray(name),query_index=np.empty(0,int),frame_index=np.empty(0,int),dense_evidence=np.empty((0,3,32,32),np.float32),candidate_metadata=np.empty((0,4),np.float32),native_margin=np.empty(0),outcome=np.empty(0,int),candidate_error=np.empty(0),gt_visible=np.empty(0,bool))
    table=pd.concat([pd.read_csv(x) for x in native],ignore_index=True)
    if table.segment_name.duplicated().any():raise ValueError('Duplicate native video')
    required=['segment_name','eval_rows','native_valid','C','L','A','native_TP','native_TN','native_FP','native_FN','native_average_jaccard','native_average_pts_within_thresh','native_occlusion_accuracy']+[f'native_jaccard_{r}' for r in (1,2,4,8,16)]
    if set(table.segment_name)!=set(read_ids(a.split)):raise ValueError('Native videos do not match selected split')
    table[required].to_csv(a.output/'native.csv',index=False)
    index(a.output/'evidence',a.split,a.output/'folds.csv')
    print('PREPARED',a.host,a.output)
if __name__=='__main__':main()
