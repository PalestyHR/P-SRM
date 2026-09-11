"""Frozen TAPNext++ source/target export using the project's validated decoder and adapter."""
from pathlib import Path
import argparse,csv,json,pickle,sys,time
import numpy as np
import torch
import mediapy as media
from .tapnext_model import load_model,infer_sequence,decode_example,official_axis_distribution,candidate_centered_evidence
from psrm.data import build_dense_projection

ROOT=DATA=REPO=SPLITS=CHECKPOINT=DAVIS=None

def examples(split):
    if split=='source':
        with (SPLITS).open() as f:wanted={r['segment'] for r in csv.DictReader(f)}
        order=[]
        with (DATA/'tapvid_kinetics_development.csv').open() as f:
            for r in csv.reader(f):
                s=f'{r[0]}_{int(r[1]):06d}_{int(r[2]):06d}'
                if s not in order:order.append(s)
        idx=0
        for shard in sorted((DATA/'pickle_development_10').glob('*.pkl')):
            with shard.open('rb') as f: data=pickle.load(f)
            for e in data:
                name=order[idx];idx+=1
                if name in wanted:yield name,e
            del data
    else:
        with (DAVIS).open('rb') as f:data=pickle.load(f)
        yield from data.items()

def decode(e,split):
    if split=='source':return decode_example(e)
    frames=np.asarray(media.resize_video(e['video'],(256,256)),np.float32)/127.5-1
    xy=np.asarray(e['points'],np.float32)*256;occ=np.asarray(e['occluded'],bool)
    keep=(~occ).any(1);xy=xy[keep];occ=occ[keep]
    q=[]
    for i in range(len(xy)):
        t=int(np.flatnonzero(~occ[i])[0]);q.append([t,xy[i,t,1],xy[i,t,0]])
    return frames,xy,occ,np.asarray(q,np.float32)

def metrics(name,xy,vis,gt,occ,ev,outcome):
    error=np.linalg.norm(xy-gt,axis=-1);correct=(error<=8)&~occ
    tp=int((ev&vis&correct).sum());fp=int((ev&vis&~correct).sum())
    fn=int((ev&~occ&~(vis&correct)).sum());tn=int((ev&occ&~vis).sum())
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    row=dict(segment_name=name,eval_rows=int(ev.sum()),native_valid=int((ev&vis).sum()),
      C=int((outcome==0).sum()),L=int((outcome==1).sum()),A=int((outcome==2).sum()),
      native_TP=tp,native_TN=tn,native_FP=fp,native_FN=fn,native_precision=p,native_recall=r,
      native_f1=2*p*r/max(p+r,1e-30),native_accuracy=(tp+tn)/max(tp+tn+fp+fn,1),
      native_occlusion_accuracy=float(((vis==~occ)&ev).sum()/ev.sum()))
    aj=[];pts=[];nvis=int((ev&~occ).sum())
    for k in [1,2,4,8,16]:
        good=(error<k)&~occ;n=int((ev&vis&good).sum());false=int((ev&vis&~good).sum())
        row[f'native_jaccard_{k}']=n/max(nvis+false,1);aj.append(row[f'native_jaccard_{k}'])
        pts.append(float((ev&good).sum()/max(nvis,1)))
    row['native_average_jaccard']=float(np.mean(aj));row['native_average_pts_within_thresh']=float(np.mean(pts))
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['source','davis'],required=True);ap.add_argument('--max-videos',type=int)
    global ROOT,DATA,REPO,SPLITS,CHECKPOINT,DAVIS
    ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--tapnet-root',type=Path,required=True);ap.add_argument('--folds',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--davis',type=Path)
    args=ap.parse_args();ROOT=args.output_root;DATA=args.data_root;REPO=args.tapnet_root;SPLITS=args.folds;CHECKPOINT=args.checkpoint;DAVIS=args.davis
    out=ROOT
    for name in ['packets','history','raw','native']: (out/name).mkdir(parents=True,exist_ok=True)
    model=load_model(REPO,CHECKPOINT,torch.device('cuda'))
    with (SPLITS).open() as f:folds={r['segment']:int(r['fold']) for r in csv.DictReader(f)}
    done=0
    for name,e in examples(args.split):
        if (out/'native'/f'{name}.json').exists():continue
        if args.max_videos is not None and done>=args.max_videos:break
        begin=time.time();frames,gt,occ,queries=decode(e,args.split)
        xy,logits,visible=infer_sequence(model,frames,queries,torch.device('cuda'),True)
        valid=visible>0;ev=np.arange(len(frames))[None,:]>queries[:,0:1]
        qi,fi=np.nonzero(ev&~valid);cand=xy[qi,fi];visible_gt=~occ[qi,fi]
        err=np.linalg.norm(cand-gt[qi,fi],axis=1).astype(np.float32)
        outcome=np.where(~visible_gt,2,np.where(err<=8,0,1)).astype(np.int8)
        ylog,xlog=np.split(logits,2,axis=-1)
        task,mask=candidate_centered_evidence(official_axis_distribution(xlog[qi,fi]),official_axis_distribution(ylog[qi,fi]),cand)
        dense,meta=build_dense_projection(task,cand,np.full_like(cand,15.5));dense[:,2]=mask
        np.savez_compressed(out/'packets'/f'{name}.npz',segment_name=np.asarray(name),dense_evidence=dense,candidate_metadata=meta,
          query_index=qi.astype(np.int32),frame_index=fi.astype(np.int32),native_margin=visible[qi,fi].astype(np.float32),
          outcome=outcome,candidate_error=err,gt_visible=visible_gt,source_id=np.asarray([f'{name}:{q}:{f}' for q,f in zip(qi,fi)]))
        # Keep the native outputs and axis evidence so neither source nor target needs another forward.
        np.savez_compressed(out/'raw'/f'{name}.npz',tracks_xy=xy,visible_logits=visible,track_logits_x=xlog,track_logits_y=ylog,
          gt_xy=gt,gt_occluded=occ,evaluation_mask=ev,query_points_tyx=queries)
        hq,hf=np.nonzero(ev)
        np.savez_compressed(out/'history'/f'{name}.npz',segment_name=np.asarray(name),query_index=hq.astype(np.int32),
          frame_index=hf.astype(np.int32),query_frame=queries[hq,0].astype(np.int32),candidate_xy=xy[hq,hf],
          native_margin=visible[hq,hf],native_visible_score=1/(1+np.exp(-np.clip(visible[hq,hf],-60,60))),native_valid=valid[hq,hf])
        row=metrics(name,xy,valid,gt,occ,ev,outcome)
        row.update(fold=int(folds[name]) if args.split=='source' else -1,rows=len(qi),positives=int((outcome==0).sum()))
        (out/'native'/f'{name}.json').write_text(json.dumps(row),encoding='utf-8')
        print(json.dumps(dict(segment=name,split=args.split,rejects=len(qi),seconds=round(time.time()-begin,2))),flush=True);done+=1
    rows=[json.loads(p.read_text()) for p in sorted((out/'native').glob('*.json'))]
    with (out/'native_per_video.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'folds.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['segment','rows','positives','fold']);w.writeheader()
        w.writerows([dict(segment=r['segment_name'],rows=r['rows'],positives=r['positives'],fold=r['fold']) for r in rows])
    print('EXPORT_FINISHED',args.split,len(rows),flush=True)
if __name__=='__main__':main()
