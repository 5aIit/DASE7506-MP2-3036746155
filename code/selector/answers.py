"""Answers for the frozen selector."""

import numpy as np

def aggregate_scores(scores, answers, mode):
    """Compare equal answers without awarding duplicated candidates extra weight."""
    if mode == 'candidate':
        return scores
    result = np.empty_like(scores)
    for i, row in enumerate(answers):
        for j, answer in enumerate(row):
            group = [k for k, other in enumerate(row) if answer is not None and other == answer]
            result[i, j] = scores[i, group].mean() if group else scores[i, j]
    return result

def choose(scores, answers):
    """Never select an unparseable answer when a valid candidate exists."""
    result = np.asarray(scores).copy()
    for i, row in enumerate(answers):
        if any(answer is not None for answer in row):
            result[i, [j for j, answer in enumerate(row) if answer is None]] = -np.inf
    return [int(j) for j in result.argmax(axis=1)] if len(result) else []
