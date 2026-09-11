"""Train the paper recipe or evaluate a released model on prepared host evidence."""
from pathlib import Path
import argparse,json,subprocess,sys,shutil
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from .runtime import RecoveryModel
from .prepare_dataset import evaluation_records
from .point_external import load_scores,load_s3,join_s3,fit_balanced,evaluate_fixed
from .point_metrics import crossfit_logistic,scan_system
from .tracknet_external import align_to_candidates
from .tracknet_metrics import load_sequences,select_threshold,summarize
from .tracknet_oof import bootstrap_pooled_delta

def run(module,*arguments):
    subprocess.run([sys.executable,'-m','psrm.'+module,*map(str,arguments)],check=True)

def parameters(scaler,model):
    return dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coefficient=model.coef_[0].tolist(),intercept=float(model.intercept_[0]))

def calibrate(config,data,run_root,seed):
    q,h=join_s3(load_scores(run_root/f'quality/seed_{seed}/oof_scores_all.npz'),load_s3(data/'train/history'))
    hs,hm=fit_balanced(np.column_stack([q.quality_logit.to_numpy(float),h]),(q.outcome==0).to_numpy())
    stacked=load_scores(run_root/'history/ANCHOR_PLUS_S3/oof_scores_all.npz')
    # This is the separately cross-fitted second stage, never an in-sample score.
    stacked['score_final']=crossfit_logistic(stacked,['quality_logit','native_margin'])
    ms,mm=fit_balanced(stacked[['quality_logit','native_margin']].to_numpy(float),(stacked.outcome==0).to_numpy())
    if config['task']=='point':
        selected,_,_=scan_system(stacked.rename(columns={'segment':'segment_name'}),'score_final',pd.read_csv(data/'train/native.csv'))
        threshold=selected['threshold']
    else:
        source=load_sequences(data/'train/csv');stacked['source_id']=stacked.segment+':'+stacked.frame_index.astype(str)
        aligned=align_to_candidates(source,stacked.drop(columns=['candidate_error','gt_visible'],errors='ignore'))
        threshold,_,_=select_threshold(source,aligned.score_final.to_numpy())
    bundle=run_root/'model';bundle.mkdir(exist_ok=True)
    payload=torch.load(run_root/'refit/quality.pt',map_location='cpu',weights_only=True)
    torch.save({'model':payload['model'],'config':payload['config']},bundle/'quality.pt')
    spec=dict(format_version=1,history=parameters(hs,hm),margin=parameters(ms,mm),threshold=float(threshold),history_logit_dtype='float64',score_dtype='float64',seed=seed,scope='full training split')
    (bundle/'readout.json').write_text(json.dumps(spec,indent=2)+'\n')
    return bundle

def evaluate(config,data,bundle,output,device,batch_size=512):
    model=RecoveryModel(bundle,device);pieces=[]
    hist=load_s3(data/'history')
    for record in evaluation_records(data/'evidence'):
        with np.load(record.path,allow_pickle=False) as z:
            frame=pd.DataFrame(dict(segment=record.segment,query_index=z['query_index'],frame_index=z['frame_index'],native_margin=z['native_margin'].astype(float),outcome=z['outcome'],candidate_error=z['candidate_error'],gt_visible=z['gt_visible']))
            aligned,h=join_s3(frame,hist[hist.segment==record.segment])
            values=[];dense=z['dense_evidence'];metadata=z['candidate_metadata']
            for start in range(0,len(frame),batch_size):
                end=start+batch_size
                values.append(model.score(dense[start:end],metadata[start:end],h[start:end],frame.native_margin.to_numpy()[start:end]))
            aligned['score_final']=np.concatenate(values);pieces.append(aligned)
    frame=pd.concat(pieces,ignore_index=True)
    if config['task']=='point':
        summary,per,_=evaluate_fixed(frame,pd.read_csv(data/'native.csv'),frame.score_final.to_numpy(),model.threshold)
    else:
        native=load_sequences(data/'csv');frame['source_id']=frame.segment+':'+frame.frame_index.astype(str)
        aligned=align_to_candidates(native,frame.drop(columns=['candidate_error','gt_visible'],errors='ignore'))
        summary,per=summarize(native,aligned.score_final.to_numpy()>=model.threshold)
        summary.update(bootstrap_pooled_delta(per))
    summary.update(AP_r=float(average_precision_score(frame.outcome==0,frame.score_final)),native_AP_r=float(average_precision_score(frame.outcome==0,frame.native_margin)),threshold=float(model.threshold))
    output.mkdir(parents=True,exist_ok=True);(output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n');per.to_csv(output/'per_video.csv',index=False)
    print(json.dumps(summary,indent=2));return summary

def train(a,config):
    data=a.data;out=a.output;out.mkdir(parents=True,exist_ok=True)
    common=['--evidence-root',data/'train/evidence','--fold-manifest',data/'train/folds.csv','--seed',a.seed,'--device',a.device]
    run('train_bce',*common,'--output-root',out/'bce','--aux-mode',a.auxiliary,'--presence-weight',.5,'--epochs',10,'--schedule-epochs',10,'--batch-size',256,'--learning-rate',3e-4,'--weight-decay',1e-4,'--warmup-fraction',.05)
    if a.task_kl == 'on':
        run('train_pauc',*common,'--pretrained-root',out/'bce','--output-root',out/'quality','--method','kl','--head-policy','preserve','--epochs',1,'--learning-rate',3e-6,'--weight-decay',2e-4,'--sampling-rate',.5,'--batch-size',256,'--sampler',config['sampler'])
    else:
        for fold in range(5):
            source=out/f'bce/seed_{a.seed}/fold_{fold}';dest=out/f'quality/seed_{a.seed}/fold_{fold}';dest.mkdir(parents=True,exist_ok=True)
            z=torch.load(source/'final_epoch.pt',map_location='cpu',weights_only=True)
            torch.save({'model':z['model'],'config':z['config']},dest/'final_model.pt')
            shutil.copyfile(source/'oof_scores.npz',dest/'oof_scores.npz')
        shutil.copyfile(out/f'bce/seed_{a.seed}/oof_scores_all.npz',out/f'quality/seed_{a.seed}/oof_scores_all.npz')
    run('fit_history','--evidence-root',data/'train/evidence','--fold-manifest',data/'train/folds.csv','--anchor-root',out/'quality','--s3-root',data/'train/history','--output-root',out/'history','--seed',a.seed,'--device',a.device)
    if config['task']=='box':
        run('kcf_metrics','--oof',out/'history/ANCHOR_PLUS_S3/oof_scores_all.npz','--corpus-root',data/'train','--output-root',out/'evaluation')
        return
    run('refit','--train-evidence-root',data/'train/evidence','--train-manifest',data/'train/folds.csv','--evaluation-evidence-root',data/'evaluation/evidence','--evaluation-manifest',data/'evaluation/folds.csv','--output-root',out/'refit','--seed',a.seed,'--sampler',config['sampler'],'--device',a.device,'--aux-mode',a.auxiliary,*(['--skip-kl'] if a.task_kl=='off' else []))
    bundle=calibrate(config,data,out,a.seed)
    evaluate(config,data/'evaluation',bundle,out/'evaluation',a.device)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--auxiliary',choices=['presence','none'],default='presence');p.add_argument('--task-kl',choices=['on','off'],default='on')
    p.add_argument('--weights',type=Path);p.add_argument('--seed',type=int,choices=[42,3407,8008],default=42);p.add_argument('--device',default='cpu')
    a=p.parse_args();config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    if a.command=='train':train(a,config)
    else:
        if config['task']=='box':p.error('KCF is grouped OOF: use the documented five-fold evaluation command.')
        if a.weights is None:p.error('--weights is required for evaluate')
        evaluate(config,a.data,a.weights,a.output,a.device)
if __name__=='__main__':main()
