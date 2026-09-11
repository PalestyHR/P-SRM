"""Build a local video/label manifest from the published group IDs."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
from .tracknet_v3_model import read_video

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['shuttlecock','racketvision'],required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corrected-test-labels',type=Path);a=p.parse_args()
    groups=pd.read_csv(a.split);rows=[];medians={};a.output.parent.mkdir(parents=True,exist_ok=True)
    for segment in groups.segment:
        sport,match,rally=segment.split(':',2)
        if a.dataset=='racketvision':
            video=a.root/sport/'videos'/f'{match}_{rally}.mp4';labels=a.root/sport/'all'/match/'csv'/f'{rally}_ball.csv';median=a.root/sport/'all'/match/'median.npz'
        else:
            if match.startswith(('Professional_','Amateur_')):
                domain,local=match.split('_',1);base=a.root/domain/local
            elif match.startswith('Test_'):base=a.root/match[5:]
            else:base=a.root/match
            video=base/'video'/f'{rally}.mp4'
            labels=(a.corrected_test_labels/(match[5:] if match.startswith('Test_') else match)/'corrected_csv'/f'{rally}_ball.csv') if a.corrected_test_labels else base/'csv'/f'{rally}_ball.csv'
            if str(base) not in medians:
                values=[np.median(read_video(v,rgb=True),axis=0) for v in sorted((base/'video').glob('*.mp4'))]
                if not values:raise FileNotFoundError(base/'video')
                median=a.output.parent/f'{match}_median.npy';np.save(median,np.median(np.stack(values),axis=0).astype(np.uint8));medians[str(base)]=median
            median=medians[str(base)]
        for file in (video,labels,median):
            if not file.is_file():raise FileNotFoundError(file)
        rows.append(dict(segment=segment,dataset=a.dataset,sport=sport,video=str(video.resolve()),labels=str(labels.resolve()),median=str(median.resolve())))
    pd.DataFrame(rows).to_csv(a.output,index=False);print('MANIFEST',len(rows),a.output)
if __name__=='__main__':main()
