"""Training data for the frozen selector."""

import numpy as np

from sklearn.linear_model import LogisticRegression

from common import DATA, load_inputs, parse_answer, read_jsonl

from selector.features import extract_features, answer_groups

from selector.answers import choose

SEED = 7506

def load_training_split(split):
    if split not in ('train', 'validation'):
        raise ValueError('Training and selection only accept train/validation')
    bank, arrays = load_inputs(split)
    references = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    answers = answer_groups(bank)
    assert all(b['id'] == r['id'] and b['question'] == r['question']
               for b, r in zip(bank, references)) and len(bank) == len(references)
    labels = np.array([[a is not None and a == parse_answer(r['answer']) for a in row]
                       for row, r in zip(answers, references)], dtype=np.int8)
    features, names = extract_features(bank, arrays['embeddings'])
    hard = np.array([len({a for a in row if a is not None}) >= 2 for row in answers])
    return bank, arrays['embeddings'], features, names, answers, labels, hard

def metrics(scores, answers, labels, hard, mask_invalid=True):
    selected = np.array(choose(scores, answers) if mask_invalid else scores.argmax(1))
    correct = labels[np.arange(len(labels)), selected].astype(bool)
    return {'correct': int(correct.sum()), 'n': len(correct),
            'accuracy': float(100 * correct.mean()),
            'hard_correct': int(correct[hard].sum()), 'hard_n': int(hard.sum()),
            'hard_accuracy': float(100 * correct[hard].mean())}

def fit_linear(x, labels, c):
    informative = (labels.min(1) != labels.max(1))
    scale = x[informative].reshape(-1, x.shape[-1]).std(0).astype(np.float64)
    scale = np.maximum(scale, 1e-4)
    differences, weights = [], []
    for row, y in zip(x[informative], labels[informative]):
        good, bad = np.flatnonzero(y), np.flatnonzero(1-y)
        for a in good:
            for b in bad:
                differences.append((row[a] - row[b]) / scale)
                weights.append(1.0 / (len(good) * len(bad)))
    delta = np.asarray(differences, dtype=np.float64)
    pair_x = np.concatenate([delta, -delta])
    pair_y = np.concatenate([np.ones(len(delta)), np.zeros(len(delta))])
    pair_w = np.tile(weights, 2)
    estimator = LogisticRegression(C=c, fit_intercept=False, solver='lbfgs',
                                   max_iter=500, tol=1e-6, random_state=SEED)
    estimator.fit(pair_x, pair_y, sample_weight=pair_w)
    return {'type': 'linear', 'weight': (estimator.coef_[0]/scale).astype(np.float32),
            'bias': 0.0, 'iterations': int(estimator.n_iter_[0])}
