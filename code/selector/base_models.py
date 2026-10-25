"""Base models for the frozen selector."""

import json, time

import numpy as np

from catboost import CatBoostRanker

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold

from selector.meta_features import make_meta

from selector.comparison_features import meta_design, normalize

from selector.boost_features import design as boost_design

from selector.context_network import transform as context_transform, forward

from selector.polynomial_features import transform as poly_transform

from selector.pairwise_models import design as pair_design, model_scores as pair_scores

from selector.answers import aggregate_scores

SEED=20261030

def fit_components(e,f,t,a,y,h):
    from selector.training_data import fit_linear
    from selector.meta_training import oof_scores, fit_meta
    from selector.boost_training import parameters, group_pool
    from selector.context_training import fit_statistics, fit as fit_context
    from selector.group_loss import pooling
    from selector.pairwise_models import select_dimensions, pair_data, hgb_parameters
    from selector.polynomial_training import fit_transform as fit_poly
    raw=np.concatenate([e,f],axis=-1);rt=np.concatenate([e,f,t],axis=-1)
    m={'base':fit_linear(raw,y,.03)['weight'],'trace':fit_linear(rt,y,.03)['weight']}
    oof=oof_scores(raw,y,h,np.arange(len(e)))
    mx=make_meta(f,t,oof,a,True);m['meta']=fit_meta(mx,y,h,15)
    bx=boost_design(e,f,t,'context');m['boost']=CatBoostRanker(**parameters(6,10.,'PairLogit',266))
    m['boost'].fit(group_pool(bx,y,np.flatnonzero(h)))
    stats=fit_statistics(e,f,t,np.arange(len(e)));x,c=context_transform(e,f,t,stats)
    anchor=np.pad(m['base'],(0,t.shape[-1]))*stats['scale'];groups=pooling(a)
    targets=np.einsum('ngk,nk->ng',groups,y).argmax(1)
    cfg={'hidden':32,'conditional':True,'reg':.001,'dropout':0.,'epochs':40}
    w,_=fit_context(x[h],c[h],groups[h],targets[h],anchor,cfg)
    m['context']={'stats':stats,'weights':w}
    dims=select_dimensions(e,y,np.flatnonzero(h),128);x=pair_design(e,f,t,dims)
    px,py,pw=pair_data(x[h],y[h]);pair=HistGradientBoostingClassifier(**hgb_parameters(15))
    pair.fit(px,py,sample_weight=pw);m['pair']={'dims':dims,'model':pair}
    poly=fit_poly(rt,y,'pca',32);poly['weight']=fit_linear(poly_transform(rt,poly),y,.003)['weight'];m['poly']=poly
    return m

def components(e,f,t,a,m):
    raw=np.concatenate([e,f],axis=-1);rt=np.concatenate([e,f,t],axis=-1);base=raw@m['base']
    mx=make_meta(f,t,base,a,True);meta=m['meta'].decision_function(mx.reshape(-1,mx.shape[-1])).reshape(len(e),4)
    bx=boost_design(e,f,t,'context');boost=m['boost'].predict(bx.reshape(-1,bx.shape[-1]),thread_count=4).reshape(len(e),4)
    x,c=context_transform(e,f,t,m['context']['stats']);context=forward(x,c,m['context']['weights'],True)[0]
    px=pair_design(e,f,t,m['pair']['dims']);pair=pair_scores(m['pair']['model'],px,'pair_hgb')
    poly=poly_transform(rt,m['poly'])@m['poly']['weight'];trace=rt@m['trace']
    return np.stack([base,meta,boost,context,pair,poly,trace],axis=-1).astype(np.float32)

def crossfit(e,f,t,a,y,h,label):
    scores=np.zeros((len(e),4,7),dtype=np.float32)
    for fold,(fit,hold) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(e,h)):
        tick=time.perf_counter()
        m=fit_components(e[fit],f[fit],t[fit],[a[i] for i in fit],y[fit],h[fit])
        scores[hold]=components(e[hold],f[hold],t[hold],[a[i] for i in hold],m)
        print(json.dumps({'stage':label,'fold':fold,'seconds':time.perf_counter()-tick}),flush=True)
    return scores

def design(f,t,sem,scores,answers,schema):
    out=meta_design(f,t,scores,answers,'scores' if schema=='scores' else 'full')
    return np.concatenate([out,sem],axis=-1) if schema=='semantic' else out

def incumbent(scores,a):return aggregate_scores(.5*scores[:,:,1]+.5*scores[:,:,2],a,'answer_mean')

def combine(scores,head,a,alpha):
    if not alpha:return incumbent(scores,a)
    return (1-alpha)*normalize(incumbent(scores,a))+alpha*normalize(aggregate_scores(head,a,'answer_mean'))
