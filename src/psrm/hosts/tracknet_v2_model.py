"""Execute the original checkpoint graph in PyTorch without changing layer semantics."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class SavedBatchNorm(nn.Module):
    def __init__(self,weights,axis,epsilon):
        super().__init__()
        self.axis=axis;self.epsilon=epsilon
        for name,value in zip(['gamma','beta','mean','variance'],weights):
            self.register_buffer(name,torch.from_numpy(value.copy()))
    def forward(self,x):
        shape=[1]*x.ndim;shape[self.axis]=len(self.gamma)
        return (x-self.mean.reshape(shape))*torch.rsqrt(self.variance.reshape(shape)+self.epsilon)*self.gamma.reshape(shape)+self.beta.reshape(shape)

class OriginalTrackNetV2(nn.Module):
    def __init__(self,config_path,weights_path):
        super().__init__()
        self.layers=json.loads(Path(config_path).read_text(encoding='utf-8'))['config']['layers']
        self.ops=nn.ModuleDict()
        with np.load(weights_path,allow_pickle=False) as archive:
            for layer in self.layers:
                cfg=layer['config'];name=cfg['name'];kind=layer['class_name']
                values=[archive[f'{name}/{i}'] for i in range(4) if f'{name}/{i}' in archive]
                if kind=='Conv2D':
                    kernel=values[0].transpose(3,2,0,1).copy()
                    assert cfg['padding']=='same' and cfg['data_format']=='channels_first'
                    assert tuple(cfg['strides'])==(1,1) and tuple(cfg['dilation_rate'])==(1,1)
                    op=nn.Conv2d(kernel.shape[1],kernel.shape[0],cfg['kernel_size'],padding=tuple(k//2 for k in cfg['kernel_size']),bias=cfg['use_bias'])
                    op.weight.data.copy_(torch.from_numpy(kernel))
                    if cfg['use_bias']:op.bias.data.copy_(torch.from_numpy(values[1]))
                elif kind=='BatchNormalization':op=SavedBatchNorm(values,cfg['axis'],cfg['epsilon'])
                elif kind=='Activation':
                    op={'relu':nn.ReLU,'sigmoid':nn.Sigmoid}[cfg['activation']]()
                elif kind=='MaxPooling2D':
                    assert cfg['padding']=='valid' and cfg['data_format']=='channels_first'
                    op=nn.MaxPool2d(cfg['pool_size'],stride=cfg['strides'])
                elif kind=='UpSampling2D':op=nn.Upsample(scale_factor=tuple(cfg['size']),mode='nearest')
                elif kind in ['InputLayer','Concatenate']:continue
                else:raise ValueError(f'Unsupported original layer: {kind}')
                self.ops[name]=op
    def forward(self,x):
        values={}
        for layer in self.layers:
            cfg=layer['config'];name=cfg['name'];kind=layer['class_name']
            if kind=='InputLayer':values[name]=x;continue
            incoming=[values[node[0]] for node in layer['inbound_nodes'][0]]
            values[name]=torch.cat(incoming,dim=cfg['axis']) if kind=='Concatenate' else self.ops[name](incoming[0])
        return values[self.layers[-1]['config']['name']]
