"""Meta training for the frozen selector."""

import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import HistGradientBoostingClassifier

from selector.training_data import fit_linear

SEED=20261021

def oof_scores(raw,labels,hard,indices):
    result=np.zeros((len(indices),4),dtype=np.float32)
    for fit,hold in StratifiedKFold(4,shuffle=True,random_state=SEED).split(indices,hard[indices]):
        model=fit_linear(raw[indices[fit]],labels[indices[fit]],.03)
        result[hold]=raw[indices[hold]]@model['weight']
    return result

def fit_meta(x,labels,hard,leaves):
    # Each informative question has equal total mass, split equally over good/bad.
    xx=x[hard];yy=labels[hard]
    w=np.where(yy,.5/np.maximum(1,yy.sum(1))[:,None],.5/np.maximum(1,(1-yy).sum(1))[:,None])
    m=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=leaves,learning_rate=.05,
        l2_regularization=5.,min_samples_leaf=40,early_stopping=False,random_state=SEED)
    m.fit(xx.reshape(-1,x.shape[-1]),yy.ravel(),sample_weight=w.ravel())
    return m
