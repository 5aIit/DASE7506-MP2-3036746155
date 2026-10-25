"""Polynomial features for the frozen selector."""

import numpy as np

def transform(raw, model):
    kind=model['kind']
    if kind=='baseline':return raw[...,:1108]
    shape=raw.shape[:2]
    flat=raw.reshape(-1,raw.shape[-1])
    z=np.clip((flat-model['mean'])/model['scale'],-5.,5.)
    if kind=='sketch':
        # Bounded working memory for FFT-based TensorSketch.
        phi=np.concatenate([model['sketch'].transform(z[i:i+256]).astype(np.float32)
                            for i in range(0,len(z),256)],axis=0)
    else:
        low=(z[:,:1024]-model['pca_mean'])@model['components']
        if kind=='selected':low=np.concatenate([low,z[:,model['selected_columns']]],axis=-1)
        indices=np.triu_indices(low.shape[-1])
        phi=low[:,indices[0]]*low[:,indices[1]]
    return np.concatenate([flat,phi],axis=-1).reshape(*shape,-1).astype(np.float32)
