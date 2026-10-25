"""Context training for the frozen selector."""

import numpy as np

from selector.training_data import metrics

from selector.answers import aggregate_scores

from selector.context_network import forward, objective, initialize

SEED = 20261025

def fit_statistics(e, f, t, indices):
    raw = np.concatenate([e[indices], f[indices], t[indices]], axis=-1)
    centered = raw-raw.mean(1, keepdims=True)
    return {'scale': np.maximum(centered.reshape(-1, raw.shape[-1]).std(0), 1e-4),
            'context_mean': e[indices].mean(1).mean(0),
            'context_scale': np.maximum(e[indices].mean(1).std(0), 1e-4)}

def evaluate(scores, answers, labels, hard):
    return metrics(aggregate_scores(scores, answers, 'answer_mean'), answers, labels, hard)

def fit(x, context, groups, target, anchor, config, dev=None):
    rng = np.random.default_rng(SEED)
    weights = initialize(x.shape[-1], context.shape[-1], config['hidden'], anchor, SEED)
    first = {k:np.zeros_like(v) for k,v in weights.items()}
    second = {k:np.zeros_like(v) for k,v in weights.items()}
    step = 0
    history, best, best_weights = [], None, None
    for epoch in range(1, config['epochs']+1):
        order = rng.permutation(len(x))
        for begin in range(0, len(x), 128):
            idx = order[begin:begin+128]
            drop = config['dropout']
            mask = ((rng.random((len(idx),config['hidden']))>=drop)/(1-drop)).astype(np.float32) if drop else None
            loss, grads = objective(x[idx], context[idx], weights, groups[idx], target[idx],
                                    anchor, config['reg'], config['conditional'], mask)
            step += 1
            for key, gradient in grads.items():
                first[key] = .9*first[key]+.1*gradient
                second[key] = .999*second[key]+.001*gradient*gradient
                weights[key] -= .0005*(first[key]/(1-.9**step))/(np.sqrt(second[key]/(1-.999**step))+1e-8)
        if dev is not None and epoch % 5 == 0:
            dx, dc, da, dy, dh = dev
            scores = forward(dx, dc, weights, config['conditional'])[0]
            record = {'epoch':epoch, **evaluate(scores, da, dy, dh)}
            history.append(record)
            key = (record['hard_correct'], record['correct'], -epoch)
            if best is None or key>best:
                best = key
                best_weights = {k:v.copy() for k,v in weights.items()}
    if dev is None:
        return weights, {'epochs':config['epochs']}
    selected = max(history, key=lambda r:(r['hard_correct'],r['correct'],-r['epoch']))
    return best_weights, {'epochs':selected['epoch'], 'history':history}
