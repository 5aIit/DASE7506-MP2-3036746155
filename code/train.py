"""Refit the one frozen recipe using only supplied training supervision.

The release checkpoint is not overwritten. Validation is read only after
training to check that the newly fitted model reproduces the frozen scores.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OMP_DYNAMIC'] = 'FALSE'

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from selector.training_data import load_training_split, metrics
from selector.base_models import crossfit, fit_components, design as heterogeneous_design
from selector.group_model import fit as fit_group, design as group_design
from selector.meta_training import fit_meta
from selector.reasoning_features import extract_reasoning_features
from selector.semantic_features import extract_semantic_features
from selector.stack_features import canonical, prepare, SEED
from selector.pairwise_models import weights
from selector.answers import choose
from submission import raw_scores

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/retraining')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new --output directory to preserve previous results')
    output.mkdir(parents=True)
    source = ROOT / 'checkpoint'
    config = json.loads((source / 'config.json').read_text())
    start = time.perf_counter()
    with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=4, user_api='openmp'):
        bank, embeddings, features, _, answers, labels, hard = load_training_split('train')
        trace, _ = extract_reasoning_features(bank)
        semantic, _ = extract_semantic_features(bank)
        oof = crossfit(embeddings, features, trace, answers, labels, hard, 'training_oof')
        components = fit_components(embeddings, features, trace, answers, labels, hard)
        compact = {
            'components': components,
            'group_head': fit_group(group_design(features, trace, semantic, oof, answers, 'semantic'),
                                    labels, hard, {'kind': 'candidate', 'leaves': 15}),
            'hetero_head': fit_meta(heterogeneous_design(features, trace, semantic, oof, answers, 'scores'),
                                    labels, hard, 7),
        }
        ordered_bank, ordered_embeddings, order = canonical(bank, embeddings)
        ordered_labels = labels[np.arange(len(labels))[:, None], order]
        ordered_oof = oof[np.arange(len(labels))[:, None], order]
        designs, _, _ = prepare(ordered_bank, ordered_embeddings, compact, ordered_oof)

        def fit_head(settings):
            x = designs[settings['schema']][hard]
            head = HistGradientBoostingClassifier(max_iter=settings['trees'],
                max_leaf_nodes=settings['leaves'], min_samples_leaf=settings['min_leaf'],
                l2_regularization=settings['l2'], learning_rate=.05,
                early_stopping=False, random_state=SEED)
            head.fit(x.reshape(-1, x.shape[-1]), ordered_labels[hard].ravel(),
                     sample_weight=weights(ordered_labels[hard]).ravel())
            return head

        compact['stable_head'] = fit_head({'schema': 'stable', 'trees': 200,
            'leaves': 15, 'min_leaf': 20, 'l2': 20.})
        model = {'base': compact, 'heads': [fit_head(h['config']) for h in config['ensemble']['heads']]}
        checkpoint = output / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'config.json').write_bytes((source / 'config.json').read_bytes())
        joblib.dump(model, checkpoint / 'model.joblib', compress=0)
        training_seconds = time.perf_counter() - start
        vb, ve, _, _, va, vy, vh = load_training_split('validation')
        frozen = raw_scores(vb, {'embeddings': ve}, source)
        reproduced = raw_scores(vb, {'embeddings': ve}, checkpoint)
        difference = float(np.max(np.abs(frozen - reproduced)))
        identical = choose(frozen, va) == choose(reproduced, va)
        result = {'all_validation_choices_identical': identical,
            'validation': metrics(reproduced, va, vy, vh),
            'max_abs_score_difference': difference, 'training_seconds': training_seconds,
            'total_seconds': time.perf_counter() - start, 'test_evaluated': False,
            'all_learned_weights_refitted': True, 'no_validation_fitting_or_selection': True}
        (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
        assert identical and difference == 0, 'Retrained scores do not exactly reproduce this environment'


if __name__ == '__main__':
    main()
