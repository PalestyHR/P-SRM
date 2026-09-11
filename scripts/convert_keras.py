"""Convert user-supplied original Keras TrackNet weights to numerical arrays."""
from pathlib import Path
import argparse,json,os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',type=int,choices=[1,2],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--architecture',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np
    from tensorflow import keras
    if a.version==1:
        if a.architecture is None:p.error('V1 requires --architecture adapters/tracknet/v1_model.json')
        model=keras.models.model_from_json(a.architecture.read_text());model.load_weights(a.checkpoint)
    else:model=keras.models.load_model(a.checkpoint,compile=False)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'model.json').write_text(model.to_json())
    np.savez(a.output/'weights.npz',**{f'{layer.name}/{i}':value for layer in model.layers for i,value in enumerate(layer.get_weights())})
    print('CONVERTED',a.output)
if __name__=='__main__':main()
