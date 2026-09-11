"""Minimal synthetic example of the candidate-preserving recovery interface."""
from pathlib import Path
import argparse,numpy as np
from psrm import RecoveryModel,CausalHistory
from psrm.data import build_dense_projection

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,default=Path('weights/tapnet'));p.add_argument('--device',default='cpu');a=p.parse_args()
    model=RecoveryModel(a.weights,a.device)
    positions=np.asarray([[64.,64.],[100.,80.],[128.,128.]],np.float32);native_valid=np.asarray([True,False,False])
    selected=positions[~native_valid];yy,xx=np.indices((32,32));evidence=np.stack([np.exp(-((xx-x/8)**2+(yy-y/8)**2)/4) for x,y in selected]).astype(np.float32)
    dense,metadata=build_dense_projection(evidence,selected,selected/8)
    history=[]
    for xy in selected:
        state=CausalHistory(0);state.update(1,xy-np.array([2,0],np.float32),.5,.7,True);state.update(2,xy-np.array([1,0],np.float32),.4,.65,True)
        history.append(state.update(3,xy,-.1,.45,False))
    final_positions,final_valid=model.recover(positions,native_valid,dense,metadata,np.asarray(history),np.array([-.1,-.1]))
    np.testing.assert_array_equal(final_positions,positions)
    assert np.all(final_valid[native_valid])
    print('P-SRM QUICK START OK: 3 candidates; original coordinates and native acceptances preserved.')
    print('Final validity:',final_valid.tolist())
if __name__=='__main__':main()
