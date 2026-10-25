"""Polynomial training for the frozen selector."""

import numpy as np

from sklearn.decomposition import PCA

from sklearn.kernel_approximation import PolynomialCountSketch

from selector.training_data import fit_linear

SEED=20261024

def fit_transform(raw,labels,kind,dim):
    model={'kind':kind}
    if kind=='baseline':return model
    informative=labels.min(1)!=labels.max(1)
    flat=raw[informative].reshape(-1,raw.shape[-1])
    model['mean']=flat.mean(0).astype(np.float32)
    model['scale']=np.maximum(flat.std(0),1e-4).astype(np.float32)
    z=np.clip((flat-model['mean'])/model['scale'],-5.,5.)
    if kind=='sketch':
        model['sketch']=PolynomialCountSketch(degree=2,gamma=1/z.shape[-1],
            coef0=0,n_components=dim,random_state=SEED).fit(z[:4])
    else:
        pca=PCA(n_components=dim,svd_solver='randomized',random_state=SEED)
        pca.fit(z[:,:1024]);model['pca_mean']=pca.mean_.astype(np.float32)
        model['components']=(pca.components_.T/np.sqrt(np.maximum(pca.explained_variance_,1e-6))).astype(np.float32)
        if kind=='selected':
            base=fit_linear(raw,labels,.03)
            importance=np.abs(base['weight'][1024:]*model['scale'][1024:])
            model['selected_columns']=(np.argsort(importance)[-32:]+1024).astype(np.int32)
    return model
