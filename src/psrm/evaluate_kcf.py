"""Score KCF with the five released models, then apply its grouped-OOF protocol."""
from pathlib import Path
import argparse,numpy as np,pandas as pd,torch
from .runtime import RecoveryModel,_affine,_sigmoid
from .point_external import load_s3,join_s3
from .data import iter_video_batches,load_fold_manifest
from .kcf_metrics import evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=load_fold_manifest(a.data/'folds.csv',a.data/'evidence');history=load_s3(a.data/'history');pieces=[]
    for fold in range(5):
        model=RecoveryModel(a.weights/f'fold_{fold}',a.device)
        for batch in iter_video_batches([r for r in records if r.fold==fold],512,seed=42,epoch=0,shuffle=False):
            frame=pd.DataFrame({k:batch[k] for k in ['segment','query_index','frame_index','native_margin','outcome','candidate_error','gt_visible','fold']})
            frame,h=join_s3(frame,history)
            with torch.inference_mode():q=model.network(torch.from_numpy(batch['dense']).to(a.device),torch.from_numpy(batch['metadata']).to(a.device)).cpu().numpy()
            hidden=_affine(np.column_stack([q,h]),model.spec['history']).astype(np.float32)
            frame['quality_logit']=hidden;frame['quality_score']=_sigmoid(hidden).astype(np.float32);pieces.append(frame)
    frame=pd.concat(pieces,ignore_index=True);path=a.output/'oof_scores.npz';np.savez_compressed(path,**{c:frame[c].to_numpy() for c in frame})
    evaluate(path,a.data,a.output/'evaluation')
if __name__=='__main__':main()
