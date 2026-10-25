"""Boost training for the frozen selector."""

import numpy as np

from catboost import Pool

SEED=20261022

def group_pool(x,y,ids):
    xx=x[ids];yy=y[ids]
    return Pool(xx.reshape(-1,xx.shape[-1]),yy.ravel(),group_id=np.repeat(np.arange(len(ids)),4))

def parameters(depth,l2,loss,iterations=600):
    return dict(iterations=iterations,depth=depth,l2_leaf_reg=l2,learning_rate=.04,
                loss_function=loss,eval_metric='NDCG:top=1',random_seed=SEED,
                thread_count=4,rsm=.3,bootstrap_type='Bernoulli',subsample=.8,
                allow_writing_files=False,verbose=False,border_count=64)
