"""Pairwise models for the frozen selector."""

import os

import numpy as np

os.environ['OMP_NUM_THREADS'] = '4'

SEED = 20261029

def select_dimensions(e, labels, fit, count):
    """Rank dimensions by standardized within-question good/bad difference."""
    ee = e[fit].astype(np.float64)
    yy = labels[fit].astype(np.float64)
    good = (ee * yy[..., None]).sum(1) / yy.sum(1)[:, None]
    bad = (ee * (1-yy)[..., None]).sum(1) / (1-yy).sum(1)[:, None]
    delta = good-bad
    importance = np.abs(delta.mean(0)) / np.maximum(delta.std(0), 1e-5)
    return np.argsort(-importance, kind='stable')[:count].astype(np.int32)

def design(e, f, trace, dims):
    ee = e[..., dims]
    ft = np.concatenate([f, trace], axis=-1)
    return np.concatenate([ee, ee-ee.mean(1, keepdims=True),
                           ft, ft-ft.mean(1, keepdims=True)], axis=-1).astype(np.float32)

def weights(labels):
    return np.where(labels, .5/np.maximum(1, labels.sum(1))[:, None],
                    .5/np.maximum(1, (1-labels).sum(1))[:, None])

def pair_data(x, labels):
    xx, yy, ww = [], [], []
    for row, y in zip(x, labels):
        good, bad = np.flatnonzero(y), np.flatnonzero(1-y)
        for a in good:
            for b in bad:
                delta = row[a]-row[b]
                xx.extend([delta, -delta]); yy.extend([1, 0])
                ww.extend([1/(len(good)*len(bad))]*2)
    return np.asarray(xx, dtype=np.float32), np.asarray(yy), np.asarray(ww)

def hgb_parameters(leaves):
    return dict(max_iter=250, max_leaf_nodes=leaves, learning_rate=.05,
                l2_regularization=10., min_samples_leaf=30,
                early_stopping=False, random_state=SEED)

def model_scores(model, x, family):
    if family == 'pair_hgb':
        delta = x[:, :, None, :]-x[:, None, :, :]
        pair = model.decision_function(delta.reshape(-1, x.shape[-1])).reshape(len(x), 4, 4)
        return ((pair-pair.transpose(0, 2, 1))/2).mean(2)
    flat = x.reshape(-1, x.shape[-1])
    if family == 'extra_trees':
        p = np.clip(model.predict_proba(flat)[:, 1], 1e-5, 1-1e-5)
        return np.log(p/(1-p)).reshape(len(x), 4)
    return model.decision_function(flat).reshape(len(x), 4)
