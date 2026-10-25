"""Boost features for the frozen selector."""

import numpy as np

def design(e,f,trace,kind):
    if kind=='raw':return np.concatenate([e,f],axis=-1)
    if kind=='context':return np.concatenate([e,e-e.mean(1,keepdims=True),f,trace],axis=-1)
    if kind=='relative':return np.concatenate([e-e.mean(1,keepdims=True),f,trace],axis=-1)
    raise ValueError('Unknown boost design')
