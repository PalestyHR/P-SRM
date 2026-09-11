"""Export TrackNetV1/V2/V3 evidence using the paper's unchanged native decoders."""
from pathlib import Path
import argparse,json,math
import cv2,numpy as np,pandas as pd,torch
from PIL import Image
from ..crep_canonicalizer import canonicalize_crep
from ..history_batch import derive_for_query
from ..prepare_dataset import packet_filename,index
from .tracknet_v1_model import OriginalTrackNetV1
from .tracknet_v2_model import OriginalTrackNetV2
from .tracknet_v3_model import read_video,load_official8,prepare_official8,infer,native_decode as decode_v3

def preprocess_v2(frame):
    x=frame.astype(np.float32);x-=x.min()
    if x.max()!=0:x/=x.max()
    image=Image.fromarray((x*255).astype(np.uint8),'RGB').resize((512,288))
    return np.asarray(image).transpose(2,0,1).astype(np.float32)/255

def decode_v2(raw,width,height):
    mask=(raw>.5).astype(np.uint8)*255
    if not mask.any():return False,0.,0.
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h=max([cv2.boundingRect(c) for c in contours],key=lambda r:r[2]*r[3]);ratio=height/288
    return True,int(ratio*(x+w/2))/ratio,int(ratio*(y+h/2))/ratio

def decode_v1(classes,width,height):
    heatmap=cv2.resize(classes,(width,height));_,mask=cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)
    circles=cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2,maxRadius=7)
    if circles is not None and len(circles)==1:return True,int(circles[0][0][0])*512/width,int(circles[0][0][1])*288/height
    return False,0.,0.

def export(a):
    manifest=pd.read_csv(a.manifest).fillna('');out=a.output
    for d in ['evidence','history','csv']:(out/d).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if a.version<3:
        cls=OriginalTrackNetV1 if a.version==1 else OriginalTrackNetV2
        model=cls(a.model_config,a.checkpoint).to(a.device).eval()
    else:model=load_official8(a.tracknet_root,a.checkpoint,a.device)
    for item in manifest.to_dict('records'):
        segment=item['segment'];frames=read_video(Path(item['video']),rgb=a.version!=1);height,width=frames.shape[1:3]
        labels=pd.read_csv(item['labels']).sort_values('Frame').fillna(0)
        start=7 if item['dataset']=='racketvision' or a.version==3 else 2 if a.version==1 else 0
        limit=len(frames)//3*3 if a.version==2 else len(frames)
        labels=labels[(labels.Frame>=start)&(labels.Frame<limit)]
        if labels.empty:raise ValueError('No evaluation frames: '+segment)
        maps={}
        if a.version==3:
            mp=Path(item['median'])
            if mp.suffix=='.npz':
                with np.load(mp,allow_pickle=False) as z:median=z['median'][...,::-1]
            else:median=np.load(mp,allow_pickle=False)
            ids=labels.Frame.to_numpy(int)
            for st in range(0,len(ids),8):
                batch=ids[st:st+8];raws=infer(model,prepare_official8(frames,batch,median),'official8',a.device,8)
                maps.update(zip(batch,raws))
        elif a.version==2:
            with torch.inference_mode():
                for st in sorted({int(f)//3*3 for f in labels.Frame}):
                    x=np.concatenate([preprocess_v2(frames[j]) for j in range(st,st+3)],axis=0)[None]
                    raws=model(torch.from_numpy(x).to(a.device)).cpu().numpy()[0]
                    maps.update((st+i,raws[i]) for i in range(3))
        records=[];dense=[];meta=[];rejects=[];margins=[];outcomes=[];errors=[];visibility=[]
        for label in labels.itertuples(index=False):
            fid=int(label.Frame)
            if a.version==1:
                x=np.concatenate([cv2.resize(frames[j],(640,360)).astype(np.float32) for j in [fid,fid-1,fid-2]],axis=2).transpose(2,0,1)[None]
                with torch.inference_mode():classes=model(torch.from_numpy(x).to(a.device)).argmax(-1).reshape(360,640).to(torch.uint8).cpu().numpy()
                valid,nx,ny=decode_v1(classes,width,height);raw=classes.astype(np.float32)/255
            else:
                raw=maps[fid];valid,nx,ny=decode_v2(raw,width,height) if a.version==2 else decode_v3(raw)
            ry,rx=np.unravel_index(np.argmax(np.asarray(raw,np.float64)),raw.shape);factor=.8 if a.version==1 else 1.;cx,cy=float(rx)*factor,float(ry)*factor
            peak=float(raw.max());margin=peak-(127/255 if a.version==1 else .5)
            visible=bool(label.Visibility) and not (label.X==0 and label.Y==0)
            lw,lh=(1920,1080) if item['dataset']=='racketvision' else (width,height)
            gx,gy=float(label.X)*512/lw,float(label.Y)*288/lh
            ce=math.hypot(cx-gx,cy-gy) if visible else math.nan;ne=math.hypot(nx-gx,ny-gy) if valid and visible else math.nan
            correct=visible and ce<=4
            records.append(dict(sequence_id=segment,sport=item['sport'],frame_id=fid,gt_visible=visible,native_valid=valid,native_x=nx,native_y=ny,native_error=ne,candidate_x=cx,candidate_y=cy,candidate_error=ce,accept_label=correct,local_peak=peak,native_margin=margin))
            if not valid:
                packet=canonicalize_crep(raw,(float(rx),float(ry)),candidate_kind='point',native_margin=margin,resolution=32,rho=.25)
                dense.append(np.stack([packet.local_view.response,packet.local_view.candidate_support,packet.local_view.valid_support]));meta.append([cx/512,cy/288,float(0<=cx<512 and 0<=cy<288),0])
                rejects.append(fid);margins.append(margin);outcomes.append(0 if correct else 1 if visible else 2);errors.append(ce if visible else 0.);visibility.append(visible)
        table=pd.DataFrame(records);xy=np.where(table.native_valid.to_numpy()[:,None],table[['native_x','native_y']].to_numpy(),table[['candidate_x','candidate_y']].to_numpy())
        hf=table.frame_id.to_numpy();ids,h,_=derive_for_query(hf,xy,table.native_margin.to_numpy(),table.local_peak.to_numpy(),table.native_valid.to_numpy(),int(hf.min()))
        filename=packet_filename(segment)
        np.savez_compressed(out/'history'/filename,segment_name=np.asarray(segment),query_index=np.zeros(len(ids),np.int32),frame_index=hf[ids],s3_features=h)
        np.savez_compressed(out/'evidence'/filename,segment_name=np.asarray(segment),query_index=np.zeros(len(rejects),np.int32),frame_index=np.asarray(rejects,np.int32),dense_evidence=np.asarray(dense,np.float32).reshape(-1,3,32,32),candidate_metadata=np.asarray(meta,np.float32).reshape(-1,4),native_margin=np.asarray(margins,np.float32),outcome=np.asarray(outcomes,np.int8),candidate_error=np.asarray(errors,np.float32),gt_visible=np.asarray(visibility,bool))
        dest=out/'csv'/item['sport'];dest.mkdir(exist_ok=True);table.to_csv(dest/(Path(filename).stem+'.csv'),index=False)
        print('EXPORTED',segment,len(rejects),flush=True)
    index(out/'evidence',a.split,out/'folds.csv')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',type=int,choices=[1,2,3],required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--model-config',type=Path);p.add_argument('--tracknet-root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.version<3 and a.model_config is None:p.error('--model-config is required for V1/V2')
    if a.version==3 and a.tracknet_root is None:p.error('--tracknet-root is required for V3')
    export(a)
if __name__=='__main__':main()
