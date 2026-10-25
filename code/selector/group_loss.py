"""Group loss for the frozen selector."""

import numpy as np

def pooling(answers):
    groups = np.zeros((len(answers), 4, 4), dtype=np.float32)
    for i, row in enumerate(answers):
        for j, answer in enumerate(row):
            if answer is None or answer in row[:j]:
                continue
            members = [k for k, other in enumerate(row) if other == answer]
            groups[i, j, members] = 1/len(members)
    return groups

def loss_gradient(scores, groups, target):
    logits = np.einsum('ngk,nk->ng', groups, scores)
    valid = groups.sum(-1) > 0
    logits = np.where(valid, logits, -1e9)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    loss = -np.log(np.maximum(p[np.arange(len(p)), target], 1e-30)).mean()
    p[np.arange(len(p)), target] -= 1
    return float(loss), np.einsum('ngk,ng->nk', groups, p)/len(p)
