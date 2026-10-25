"""Comparison features for the frozen selector."""

import numpy as np

from selector.answers import aggregate_scores

def normalize(score):
    centered = score - score.mean(axis=1, keepdims=True)
    return centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-5)

def meta_design(features, trace, scores, answers, schema):
    n, _, m = scores.shape
    pieces, normalized = [], []
    for k in range(m):
        z = normalize(scores[..., k])
        grouped = normalize(aggregate_scores(z, answers, 'answer_mean'))
        rank = (z[:, :, None] >= z[:, None, :]).mean(axis=-1)
        grank = (grouped[:, :, None] >= grouped[:, None, :]).mean(axis=-1)
        gap = grouped - grouped.max(axis=1, keepdims=True)
        pieces.append(np.stack([z, grouped, z-grouped, rank, grank, gap], axis=-1))
        normalized.append(grouped)
    normalized = np.stack(normalized, axis=-1)
    pieces.append(np.stack([normalized.mean(-1), normalized.std(-1),
                           normalized.min(-1), normalized.max(-1),
                           (normalized == normalized.max(axis=1, keepdims=True)).mean(-1)], axis=-1))
    if schema == 'full':
        pieces = [features, trace] + pieces
    return np.concatenate(pieces, axis=-1).astype(np.float32)
