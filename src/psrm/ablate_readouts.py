"""Reproduce the seven point-host readout variants from trained OOF artifacts."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from .point_external import load_scores,load_s3,join_s3,fit_balanced,evaluate_fixed
from .point_metrics import scan_system
from .runtime import _sigmoid

def crossfit_matrix(x,y,fold):
    out=np.empty(len(y),float)
    for k in np.unique(fold):
        fit=fold!=k;s=StandardScaler().fit(x[fit]);m=LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=0).fit(s.transform(x[fit]),y[fit]);out[~fit]=m.predict_proba(s.transform(x[~fit]))[:,1]
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source,h=join_s3(load_scores(a.run/f'quality/seed_{a.seed}/oof_scores_all.npz'),load_s3(a.data/'train/history'))
    target,eh=join_s3(load_scores(a.run/'refit/evaluation_scores.npz'),load_s3(a.data/'evaluation/history'))
    stacked=load_scores(a.run/'history/ANCHOR_PLUS_S3/oof_scores_all.npz');keys=['segment','query_index','frame_index']
    aligned=source[keys].merge(stacked[keys+['quality_logit']],on=keys,validate='one_to_one',how='left')
    hs,hm=fit_balanced(np.column_stack([source.quality_logit,h]),(source.outcome==0).to_numpy());external_h=hm.decision_function(hs.transform(np.column_stack([target.quality_logit,eh])))
    q=source.quality_logit.to_numpy();eq=target.quality_logit.to_numpy();m=source.native_margin.to_numpy();em=target.native_margin.to_numpy();ho=aligned.quality_logit.to_numpy();y=(source.outcome==0).to_numpy();fold=source.fold.to_numpy()
    matrices={'M':(m[:,None],em[:,None]),'Q+M':(np.column_stack([q,m]),np.column_stack([eq,em])),'Full':(np.column_stack([ho,m]),np.column_stack([external_h,em])),'M+H':(np.column_stack([m,h]),np.column_stack([em,eh]))}
    native_source=pd.read_csv(a.data/'train/native.csv');native_target=pd.read_csv(a.data/'evaluation/native.csv');rows=[];a.output.mkdir(parents=True,exist_ok=True)
    for arm in ['Native','M','Q','Q+H','Q+M','Full','M+H']:
        if arm in matrices:
            x,ex=matrices[arm];score=crossfit_matrix(x,y,fold);sc,model=fit_balanced(x,y);external=model.predict_proba(sc.transform(ex))[:,1]
        elif arm=='Q':score=_sigmoid(q).astype(np.float32);external=_sigmoid(eq).astype(np.float32)
        elif arm=='Q+H':score=_sigmoid(ho).astype(np.float32);external=_sigmoid(external_h).astype(np.float32)
        else:score=m;external=em
        if arm=='Native':threshold=float('inf')
        else:
            scan=source.rename(columns={'segment':'segment_name'}).assign(score=score);selected,_,_=scan_system(scan,'score',native_source);threshold=selected['threshold']
        summary,_,_=evaluate_fixed(target,native_target,external,threshold);summary.update(arm=arm,AP_r=float(average_precision_score(target.outcome==0,external)));rows.append(summary)
    (a.output/'readout_metrics.json').write_text(json.dumps(rows,indent=2)+'\n');pd.DataFrame(rows).to_csv(a.output/'readout_metrics.csv',index=False);print('READOUT_ABLATIONS_COMPLETE',a.output)
if __name__=='__main__':main()
