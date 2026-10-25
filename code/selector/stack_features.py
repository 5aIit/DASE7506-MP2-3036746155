"""Stack features for the frozen selector."""

import numpy as np

from selector.features import extract_features, answer_groups

from selector.reasoning_features import extract_reasoning_features

from selector.semantic_features import extract_semantic_features

from selector.binding_features import extract_binding_features, boxed_transfer_features

from selector.base_models import components

from selector.answers import aggregate_scores

SEED = 20261108

def normalize(s):
    s = np.asarray(s, dtype=np.float64)
    z = s-s.mean(1, keepdims=True)
    return z/np.maximum(z.std(1, keepdims=True), 1e-5)

def canonical(bank, e):
    """Use content, never IDs, to make component arithmetic permutation-stable."""
    order = np.asarray([sorted(range(4), key=lambda j: (row['candidates'][j]['text'], e[i,j].tobytes()))
                        for i, row in enumerate(bank)], dtype=int).reshape(-1,4)
    ordered = [{**row, 'candidates': [row['candidates'][j] for j in order[i]]} for i,row in enumerate(bank)]
    return ordered, e[np.arange(len(e))[:,None],order], order

def restore(s, order):
    result = np.empty_like(s)
    result[np.arange(len(s))[:,None],order] = s
    return result

def stable_design(f,t,sem,ss,a):
    pieces=[];normalized=[]
    for k in range(ss.shape[-1]):
        z=normalize(ss[...,k]);grouped=normalize(aggregate_scores(z,a,'answer_mean'))
        for i,aa in enumerate(a):
            valid=[v for v in aa if v is not None]
            if len(valid)==len(set(valid)):
                grouped[i]=z[i]
        diff=z-grouped;diff[np.abs(diff)<1e-6]=0.
        rank=(z[:,:,None]>=z[:,None,:]).mean(-1)
        grank=(grouped[:,:,None]>=grouped[:,None,:]).mean(-1)
        pieces.append(np.stack([z,grouped,diff,rank,grank,grouped-grouped.max(1,keepdims=True)],-1))
        normalized.append(grouped)
    normalized=np.stack(normalized,-1)
    pieces.append(np.stack([normalized.mean(-1),normalized.std(-1),normalized.min(-1),normalized.max(-1),
        (normalized==normalized.max(1,keepdims=True)).mean(-1)],-1))
    sf=np.concatenate(pieces,-1)
    pooled=np.zeros((*sf.shape[:2],3*sf.shape[-1]+2),dtype=np.float64)
    for i,aa in enumerate(a):
        groups={v:[j for j,b in enumerate(aa) if b==v] for v in aa if v is not None}
        for j,v in enumerate(aa):
            ids=groups.get(v,[j]);values=sf[i,ids]
            pooled[i,j]=np.r_[values.mean(0),values.min(0),values.max(0),len(ids),len(groups)]
    return np.concatenate([f,t,sf,pooled,sem],-1).astype(np.float32)

def prepare(bank,e,m,ss=None):
    f,fn=extract_features(bank,e);t,tn=extract_reasoning_features(bank);sem,sn=extract_semantic_features(bank)
    a=answer_groups(bank)
    if ss is None:ss=components(e,f,t,a,m['components'])
    x=stable_design(f,t,sem,ss,a)
    bind,bn=extract_binding_features(bank);transfer,trn=boxed_transfer_features(bank,ss)
    return {'stable':x,'binding':np.concatenate([x,bind,transfer],-1)},a,(fn,tn,sn,bn,trn)

def blend(old,new,a,alpha):
    if not alpha:return normalize(aggregate_scores(old,a,'answer_mean'))
    ns=normalize(aggregate_scores(new,a,'answer_mean'))
    return ns if alpha==1 else (1-alpha)*normalize(aggregate_scores(old,a,'answer_mean'))+alpha*ns
