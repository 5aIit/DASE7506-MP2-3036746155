"""Context network for the frozen selector."""

import numpy as np

from selector.group_loss import loss_gradient

def transform(e, features, trace, stats):
    raw = np.concatenate([e, features, trace], axis=-1)
    x = np.clip((raw - raw.mean(1, keepdims=True)) / stats['scale'], -8, 8)
    context = np.clip((e.mean(1) - stats['context_mean']) / stats['context_scale'], -5, 5)
    return x.astype(np.float32), context.astype(np.float32)

def forward(x, context, weights, conditional=True, mask=None):
    h = np.tanh(x @ weights['U'] + weights['b'])
    if conditional:
        c = np.tanh(context @ weights['V'] + weights['a'])
    else:
        c = np.ones((len(x), h.shape[-1]), dtype=x.dtype)
    product = h * c[:, None, :]
    if mask is not None:
        product = product * mask[:, None, :]
    return x @ weights['w'] + product @ weights['v'], (h, c, product)

def objective(x, context, weights, groups, target, anchor, reg, conditional=True, mask=None):
    scores, (h, c, product) = forward(x, context, weights, conditional, mask)
    loss, ds = loss_gradient(scores, groups, target)
    dp = ds[..., None] * weights['v']
    if mask is not None:
        dp = dp * mask[:, None, :]
    dh = dp * c[:, None, :] * (1-h*h)
    gradients = {
        'w': np.einsum('nkd,nk->d', x, ds) + reg*(weights['w']-anchor),
        'v': np.einsum('nkh,nk->h', product, ds) + reg*weights['v'],
        'U': np.einsum('nkd,nkh->dh', x, dh) + reg*weights['U'],
        'b': dh.sum((0, 1)),
    }
    if conditional:
        dc = (dp*h).sum(1)*(1-c*c)
        gradients['V'] = context.T @ dc + reg*weights['V']
        gradients['a'] = dc.sum(0)
    else:
        gradients['V'] = np.zeros_like(weights['V'])
        gradients['a'] = np.zeros_like(weights['a'])
    penalty = np.sum((weights['w']-anchor)**2) + np.sum(weights['v']**2) + np.sum(weights['U']**2)
    if conditional:
        penalty += np.sum(weights['V']**2)
    return float(loss + reg*penalty/2), gradients

def initialize(d, context_d, hidden, anchor, seed):
    rng = np.random.default_rng(seed)
    return {'w': anchor.copy(), 'U': rng.normal(0, 1/np.sqrt(d), (d, hidden)).astype(np.float32),
            'V': rng.normal(0, 1/np.sqrt(context_d), (context_d, hidden)).astype(np.float32),
            'b': np.zeros(hidden, dtype=np.float32), 'a': np.zeros(hidden, dtype=np.float32),
            'v': np.zeros(hidden, dtype=np.float32)}
