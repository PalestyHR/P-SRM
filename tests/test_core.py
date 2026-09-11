from pathlib import Path
import json,numpy as np,pytest,torch
from psrm import RecoveryModel,CausalHistory,recover_candidates
from psrm.data import build_dense_projection
from psrm.history_batch import derive_for_query
from psrm.tracknet_metrics import summarize
ROOT=Path(__file__).resolve().parents[1]

def test_recovery_contract():
    positions=np.array([[1.,2.],[3.,4.],[5.,6.]])
    original=positions.copy();native=np.array([True,False,False]);scores=np.array([.8,.1])
    final,valid=recover_candidates(positions,native,scores,.5)
    np.testing.assert_array_equal(final,original);np.testing.assert_array_equal(positions,original);assert valid.tolist()==[True,True,False]
    final[0]=0;np.testing.assert_array_equal(positions,original)
    with pytest.raises(ValueError):recover_candidates(positions,native,[np.nan,.1],.5)

def test_causal_prefix_and_rejects_do_not_update_anchor():
    state=CausalHistory(0);state.update(1,np.array([1.,1.]),.3,.7,True);state.update(2,np.array([2.,1.]),.3,.7,True)
    first=state.update(3,np.array([3.,1.]),-.2,.4,False);state.update(4,np.array([500.,500.]),-.1,.4,False)
    third=state.update(5,np.array([5.,1.]),-.2,.4,False)
    assert first[17]==0 and third[17]==0 and third[5]==3 and third[6]==2
    frames=np.arange(1,9);xy=np.column_stack([frames,np.zeros(8)]).astype(np.float32);valid=np.array([1,1,0,0,1,0,1,0],bool)
    idx,h,_=derive_for_query(frames,xy,np.ones(8),np.ones(8),valid,0)
    short,hs,_=derive_for_query(frames[:5],xy[:5],np.ones(5),np.ones(5),valid[:5],0)
    np.testing.assert_array_equal(idx[:len(short)],short);np.testing.assert_array_equal(h[:len(short)],hs)

def test_projection_support():
    dense,meta=build_dense_projection(np.ones((2,32,32),np.float32),np.array([[64,64],[-1,-1]],np.float32),np.array([[8,8],[-.125,-.125]],np.float32))
    assert dense.shape==(2,3,32,32);assert dense[0,1].sum()==1;assert dense[1,1].sum()==0;assert meta[1,2]==0

@pytest.mark.parametrize('entry',json.loads((ROOT/'weights/index.json').read_text()),ids=lambda e:e['id'])
def test_released_model_load_and_kernel(entry):
    torch.set_num_threads(2);m=RecoveryModel(ROOT/'weights'/entry['id'])
    x=np.zeros((2,3,32,32),np.float32);meta=np.zeros((2,4),np.float32);h=np.zeros((2,21),np.float32);margin=np.zeros(2)
    s=m.score(x,meta,h,margin);assert s.shape==(2,) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()

def test_tracknet_localization_miss_transition():
    import pandas as pd
    frame=pd.DataFrame(dict(sequence_id=['v']*4,native_valid=[True,False,False,False],gt_visible=[True,True,True,False],native_error=[0,np.nan,np.nan,np.nan],accept_label=[True,True,False,False]))
    summary,_=summarize(frame,np.array([True,True,True]))
    assert summary['native']['TP']==1 and summary['native']['FN']==2
    assert summary['srm']['TP']==2 and summary['srm']['FP']==2 and summary['srm']['FN']==0
