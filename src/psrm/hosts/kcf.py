"""Prepare OTB2013 KCF evidence using the OpenCV observer and released labels."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np,pandas as pd
from PIL import Image
from ..kcf_adapter import project
from .box_geometry import box_iou,outcomes
from .kcf_history import causal

def prepare(a):
    labels=pd.read_csv(a.labels);folds=pd.read_csv(a.split);out=a.output
    for d in ['evidence','history','native']:(out/d).mkdir(parents=True,exist_ok=True)
    all_frames=[];manifest=[]
    special={'David':(299,770),'Football1':(0,74),'Freeman3':(0,460),'Freeman4':(0,283)}
    for group in sorted(folds.video_group.unique()):
        root=a.otb_root/group;files=sorted((root/'img').glob('*.jpg'))
        if group in special:lo,hi=special[group];files=files[lo:hi]
        annotations=[p for p in sorted(root.glob('groundtruth*.txt')) if p.read_text().strip()]
        for number,ann in enumerate(annotations):
            name=group if len(annotations)==1 else f'{group}.{number+1}'
            match=folds[folds.segment==name]
            if len(match)!=1:raise ValueError('Unlisted target sequence: '+name)
            fold=int(match.iloc[0].fold);gt=np.asarray([[float(x) for x in line.replace(',',' ').split()] for line in ann.read_text().splitlines() if line.strip()]);gt[:,:2]-=1
            if len(gt)!=len(files):raise ValueError('Frame/box length mismatch: '+name)
            native_dir=(a.native_root/name) if a.native_root else out/'native'/name
            if a.native_root is None:
                native_dir.mkdir(parents=True,exist_ok=True);frames_file=native_dir/'frames.txt';frames_file.write_text('\n'.join(str(p.resolve()) for p in files)+'\n')
                subprocess.run([str(a.observer),str(frames_file),*map(str,gt[0]),str(native_dir),'1'],check=True)
            table=pd.read_csv(native_dir/'rows.csv',float_precision='round_trip');n=len(table)
            if n!=len(files):raise ValueError('Incomplete native export: '+name)
            vis=labels[labels.sequence==name].sort_values('frame_index')
            if not np.array_equal(vis.frame_index.to_numpy(),np.arange(n)):raise ValueError('Visibility coverage mismatch: '+name)
            visible=vis.visibility.to_numpy(bool);valid=table.native_valid.to_numpy(bool)
            box=table[[f'candidate_{k}' for k in ['x','y','w','h']]].to_numpy(float);native_box=table[[f'native_{k}' for k in ['x','y','w','h']]].to_numpy(float)
            overlap=box_iou(box,gt);native_overlap=box_iou(native_box,gt);center_error=np.linalg.norm(box[:,:2]+box[:,2:]/2-gt[:,:2]-gt[:,2:]/2,axis=1)
            rejects=np.flatnonzero(~valid);dense=[];metadata=[]
            if len(rejects):
                raw=np.memmap(native_dir/'rejected_responses.f32',dtype='<f4',mode='r')
                for i in rejects:
                    row=table.iloc[i]
                    if not row.has_response:raise ValueError('Rejected row lacks same-update response')
                    h,w=int(row.response_h),int(row.response_w);st=int(row.offset_bytes)//4;response=raw[st:st+h*w].reshape(h,w)
                    roi=[row[f'roi_{k}'] for k in ['x','y','w','h']]
                    with Image.open(files[i]) as image:size=image.size
                    d,m,_=project(response,roi,bool(row.resized),box[i],size);dense.append(d);metadata.append(m)
                del raw
            scored=table.has_response.to_numpy(bool);pos=np.flatnonzero(scored)
            use=np.where(valid[:,None],native_box,box);xy=(use[:,:2]+use[:,2:]/2).astype(np.float32)
            peak=table.peak.to_numpy(np.float32);threshold=table.threshold.to_numpy(np.float32)
            ids,h=causal(pos,xy[pos],peak[pos]-threshold[pos],peak[pos],valid[pos])
            np.testing.assert_array_equal(ids,rejects)
            y=outcomes(visible[rejects].astype(int),overlap[rejects]);query=np.zeros(len(rejects),np.int32)
            # Original evidence margin subtracts in float64, then casts to float32.
            margin=(table.peak.to_numpy(float)-table.threshold.to_numpy(float))[rejects].astype(np.float32)
            np.savez_compressed(out/'evidence'/f'{name}.npz',segment_name=np.asarray(name),query_index=query,frame_index=rejects,dense_evidence=np.asarray(dense,np.float32).reshape(-1,3,32,32),candidate_metadata=np.asarray(metadata,np.float32).reshape(-1,4),native_margin=margin,candidate_box=box[rejects].astype(np.float32),candidate_error=center_error[rejects].astype(np.float32),box_iou=overlap[rejects],outcome=y,gt_visible=visible[rejects])
            np.savez_compressed(out/'history'/f'{name}.npz',segment_name=np.asarray(name),query_index=query,frame_index=rejects,s3_features=h)
            manifest.append(dict(segment=name,video_group=group,rows=len(rejects),positives=int((y==0).sum()),fold=fold,file=f'{name}.npz'))
            f=pd.DataFrame(dict(sequence=name,video_group=group,frame_index=np.arange(n),fold=fold,gt_visible=visible.astype(int),native_valid=valid.astype(int),native_iou=native_overlap,candidate_iou=overlap))
            for j,k in enumerate(['x','y','w','h']):f['native_'+k]=native_box[:,j];f['candidate_'+k]=box[:,j]
            all_frames.append(f);print('PREPARED',name,len(rejects),flush=True)
    frames=pd.concat(all_frames,ignore_index=True);rows=[]
    for group,f in frames.groupby('video_group',sort=True):
        v=f.gt_visible.to_numpy(bool);valid=f.native_valid.to_numpy(bool);good=f.native_iou.to_numpy()>=.5;r=~valid;y=outcomes(v[r].astype(int),f.candidate_iou.to_numpy()[r])
        rows.append(dict(segment_name=group,eval_rows=len(f),native_valid=int(valid.sum()),native_TP=int((valid&v&good).sum()),native_TN=int((~valid&~v).sum()),native_FP=int((valid&(~v|~good)).sum()),native_FN=int((v&(~valid|~good)).sum()),native_iou05_successes=int((valid&good).sum()),C=int((y==0).sum()),L=int((y==1).sum()),A=int((y==2).sum())))
    pd.DataFrame(manifest).to_csv(out/'folds.csv',index=False);pd.DataFrame(manifest).to_csv(out/'fold_manifest.csv',index=False);pd.DataFrame(rows).to_csv(out/'native_per_video.csv',index=False);frames.to_csv(out/'frames.csv',index=False)
    (out/'corpus.json').write_text(json.dumps({'label_source':'Qwen3-VL-2B-Instruct annotation, corrected and partially spot-checked by the authors','target_definition':'visible and fixed-box IoU >= 0.5'},indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--otb-root',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--observer',type=Path);g.add_argument('--native-root',type=Path,help='Reuse an existing output of the same observer')
    prepare(p.parse_args())
if __name__=='__main__':main()
