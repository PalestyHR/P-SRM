"""Index reader-generated evidence using the published group splits."""
from pathlib import Path
from urllib.parse import quote
import argparse,csv,json
import numpy as np
from .data import VideoRecord

def packet_filename(segment):
    return quote(str(segment),safe='-_ .')+'.npz'

def evaluation_records(root):
    records=[]
    for p in sorted(Path(root).glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            n=len(z['outcome'])
            if n: records.append(VideoRecord(str(z['segment_name']),p,n,int((z['outcome']==0).sum()),-1))
    if not records:raise ValueError('No nonempty evidence packets found')
    return records

def index(evidence,split,output):
    with Path(split).open(newline='',encoding='utf-8') as f: groups=list(csv.DictReader(f))
    folds={r['segment']:int(r.get('fold',-1)) for r in groups}
    if len(folds)!=len(groups):raise ValueError('Duplicate group in split')
    rows=[]
    for r in evaluation_records(evidence):
        if r.segment not in folds:raise ValueError('Evidence contains a group outside the selected split: '+r.segment)
        rows.append(dict(segment=r.segment,rows=r.rows,positives=r.positives,fold=folds[r.segment],file=r.path.name))
    missing=set(folds)-{r['segment'] for r in rows}
    # Zero-rejection videos still require a packet, and remain in native metrics.
    available=set()
    for p in Path(evidence).glob('*.npz'):
        with np.load(p,allow_pickle=False) as z:available.add(str(z['segment_name']))
    if missing-available:raise ValueError('Missing evidence videos: '+str(sorted(missing-available)))
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['segment','rows','positives','fold','file']);w.writeheader();w.writerows(rows)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=index(a.evidence,a.split,a.output);print(json.dumps(dict(groups=len(rows),rejected_candidates=sum(r['rows'] for r in rows))))
if __name__=='__main__':main()
