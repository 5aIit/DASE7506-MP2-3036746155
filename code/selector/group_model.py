"""Group model for the frozen selector."""

import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from catboost import CatBoostRanker, Pool

from selector.comparison_features import meta_design

from selector.pairwise_models import pair_data, weights

SEED=20261101

def design(f,t,sem,scores,answers,schema):
    score_features=meta_design(f,t,scores,answers,'scores')
    pooled=np.zeros((*scores.shape[:2],3*score_features.shape[-1]+2),dtype=np.float32)
    for i,aa in enumerate(answers):
        groups={a:[k for k,b in enumerate(aa) if a==b] for a in aa if a is not None}
        for j,a in enumerate(aa):
            ids=groups.get(a,[j]);v=score_features[i,ids]
            pooled[i,j]=np.r_[v.mean(0),v.min(0),v.max(0),len(ids),len(groups)]
    pieces=[score_features,pooled]
    if schema in ('full','semantic'):pieces=[f,t]+pieces
    if schema=='semantic':pieces.append(sem)
    return np.concatenate(pieces,axis=-1).astype(np.float32)

def fit(x,y,h,c):
    xx=x[h];yy=y[h];kind=c['kind']
    if kind in ('pair','linear'):
        tx,ty,tw=pair_data(xx,yy)
    else:tx,ty,tw=xx.reshape(-1,x.shape[-1]),yy.ravel(),weights(yy).ravel()
    if kind=='linear':
        scale=np.maximum(tx.std(0),1e-4)
        m=LogisticRegression(C=c['C'],fit_intercept=False,max_iter=500,tol=1e-6)
        m.fit(tx/scale,ty,sample_weight=tw)
        return (m.coef_[0]/scale).astype(np.float32)
    if kind=='rank':
        m=CatBoostRanker(iterations=300,depth=c['depth'],l2_leaf_reg=10.,learning_rate=.04,
            loss_function='PairLogit',rsm=.8,thread_count=4,random_seed=SEED,border_count=64,
            allow_writing_files=False,verbose=False)
        m.fit(Pool(tx,ty,group_id=np.repeat(np.arange(len(xx)),4)))
    else:
        m=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=c['leaves'],
            learning_rate=.05,min_samples_leaf=40,l2_regularization=10.,
            early_stopping=False,random_state=SEED)
        m.fit(tx,ty,sample_weight=tw)
    return m
