"""Ensemble for the frozen selector."""

import numpy as np

from selector.features import extract_features, answer_groups

from selector.reasoning_features import extract_reasoning_features

from selector.semantic_features import extract_semantic_features

from selector.binding_features import extract_binding_features, boxed_transfer_features

from selector.stack_features import canonical, normalize, stable_design, blend

from selector.head_scores import head_scores

from selector.group_model import design as group_design

from selector.comparison_features import meta_design

from selector.base_models import components, combine

from selector.answers import aggregate_scores

def preparation(bank,e,m,c,binding=False):
    b,e,order=canonical(bank,e);a=answer_groups(b)
    f,fn=extract_features(b,e);t,tn=extract_reasoning_features(b);sem,sn=extract_semantic_features(b)
    if [fn,tn,sn]!=c['feature_schemas']:raise ValueError('Wrong feature schema')
    ss=components(e,f,t,a,m['components']);x=stable_design(f,t,sem,ss,a)
    group=head_scores(m['group_head'],group_design(f,t,sem,ss,a,'semantic'))
    stable=blend(group,head_scores(m['stable_head'],x),a,c['stable_alpha'])
    hetero=combine(ss,head_scores(m['hetero_head'],meta_design(f,t,ss,a,'scores')),a,c['hetero_alpha'])
    base=sum(w*normalize(aggregate_scores(s,a,'answer_mean')) for w,s in zip(c['weights'],[stable,hetero,ss[:,:,3]]))
    xx={'stable':x}
    if binding:
        bind,_=extract_binding_features(b);tr,_=boxed_transfer_features(b,ss)
        xx['binding']=np.concatenate([x,bind,tr],-1)
    return xx,base,a,order
