"""Meta features for the frozen selector."""

import numpy as np

from selector.answers import aggregate_scores

def score_features(scores, answers):
    centered=scores-scores.mean(1,keepdims=True)
    grouped=aggregate_scores(centered,answers,'answer_mean')
    out=np.zeros((*scores.shape,10),dtype=np.float32)
    for i,row in enumerate(answers):
        valid=[j for j,a in enumerate(row) if a is not None]
        groups={a:[k for k,b in enumerate(row) if b==a] for a in row if a is not None}
        values=sorted([grouped[i,indices[0]] for indices in groups.values()],reverse=True)
        gap=values[0]-values[1] if len(values)>1 else 0
        for j,a in enumerate(row):
            others=[grouped[i,k] for k in valid if row[k]!=a]
            same=groups.get(a,[j]);sc=centered[i,same]
            out[i,j]=[centered[i,j],grouped[i,j],
                grouped[i,j]-max(others) if others else 0,
                np.mean(centered[i]<centered[i,j]),float(grouped[i,j]==max(values,default=0)),
                gap,np.std(sc),len(same),np.std(centered[i]),
                centered[i,j]-grouped[i,j]]
    return out

def make_meta(features, trace, scores, answers, with_trace):
    pieces=[features,score_features(scores,answers)]
    if with_trace:pieces.append(trace)
    return np.concatenate(pieces,axis=-1).astype(np.float32)
