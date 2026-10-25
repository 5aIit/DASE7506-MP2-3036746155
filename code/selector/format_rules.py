"""Format rules for the frozen selector."""

import numpy as np

from selector.binding_features import last_boxed_answer

from selector.features import answer_groups

def destinations(row,answers,source):
    return {last_boxed_answer(c['text']) for j,c in enumerate(row['candidates'])
        if answers[j]==source and last_boxed_answer(c['text']) is not None
        and last_boxed_answer(c['text'])!=source and last_boxed_answer(c['text']) in answers}

def transfer(bank,scores,strength):
    out=np.asarray(scores,dtype=np.float64).copy()
    if not strength:return out
    for i,(row,aa) in enumerate(zip(bank,answer_groups(bank))):
        keys=list(dict.fromkeys(a for a in aa if a is not None))
        if len(keys)<2:continue
        ids=[[j for j,a in enumerate(aa) if a==key] for key in keys]
        values=np.array([out[i,jj].mean() for jj in ids])
        top=np.flatnonzero(np.abs(values-values.max())<=1e-10)
        if len(top)!=1:continue
        source=int(top[0]);dest=destinations(row,aa,keys[source])
        if len(dest)!=1:continue
        target=keys.index(next(iter(dest)))
        probability=np.exp(values-values.max());probability/=probability.sum()
        mass=strength*probability[source];probability[source]-=mass;probability[target]+=mass
        updated=np.log(np.maximum(probability,1e-12))
        for jj,value in zip(ids,updated):out[i,jj]=value
    return out
