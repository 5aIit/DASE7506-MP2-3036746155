"""Release 1.0: offline candidate selection with the frozen learned ensemble.

Inference reads only its supplied inputs and checkpoint. The official answer
parser is shared with evaluation; reference files are never opened here.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from common import PROTOCOL
from selector.answers import aggregate_scores, choose
from selector.features import answer_groups
from selector.ensemble import preparation
from selector.head_scores import head_scores
from selector.stack_features import normalize, restore
from selector.format_rules import transfer as format_transfer
from selector.unit_rules import features as unit_features
from selector.arithmetic_rules import routes
from selector.arithmetic_repair import transfer as arithmetic_transfer
from selector.currency_rules import currency_is_unambiguous, monetary_features


def learned_scores(bank, embeddings, model, config, omit_binding=False):
    """Compare candidates once using the shared component models and heads."""
    ensemble = config['ensemble']
    features, base, answers, order = preparation(
        bank, np.asarray(embeddings, dtype=np.float32), model['base'],
        ensemble['compact'], any(h['config']['schema'] == 'binding' for h in ensemble['heads']))
    if omit_binding:
        return restore(base, order)
    total = ensemble['base_weight']
    scores = base * total
    for entry, head in zip(ensemble['heads'], model['heads']):
        member = head_scores(head, features[entry['config']['schema']])
        scores += entry['weight'] * normalize(aggregate_scores(member, answers, 'answer_mean'))
        total += entry['weight']
    return restore(scores / total, order)


def apply_rules(bank, scores, config, omit=None):
    """Apply frozen rule strengths in their original numerical order."""
    if omit != 'format_transfer':
        scores = format_transfer(bank, scores, config['ensemble']['format_transfer_strength'])
    if omit != 'target_units':
        scores = scores - unit_features(bank) @ np.asarray(config['target_unit_penalties'])
    if omit != 'fraction_repair':
        scores = arithmetic_transfer(bank, scores, config['arithmetic_transfer_strength'], routes(bank))
    production, inventory = monetary_features(bank)
    if omit != 'production_profit':
        scores = scores + production @ np.asarray(config['financial_support_weights'])
    if omit != 'inventory_profit':
        scores = scores + config['inventory_support_weight'] * inventory
    return scores


def load_checkpoint(checkpoint):
    path = Path(checkpoint)
    config = json.loads((path / 'config.json').read_text())
    if config['protocol'] != PROTOCOL or config['release'] != '1.0':
        raise ValueError('Incompatible checkpoint configuration')
    return joblib.load(path / 'model.joblib'), config


def raw_scores(bank, arrays, checkpoint):
    model, config = load_checkpoint(checkpoint)
    scores = learned_scores(bank, arrays['embeddings'], model, config)
    return apply_rules(bank, scores, config)


def predict(bank, arrays, checkpoint):
    embeddings = np.asarray(arrays['embeddings'])
    if embeddings.shape != (len(bank), 4, 1024) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid embeddings')
    if any(len(row['candidates']) != 4 for row in bank):
        raise ValueError('Expected four candidates')
    if not bank:
        return []
    with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=4, user_api='openmp'):
        return choose(raw_scores(bank, arrays, checkpoint), answer_groups(bank))
