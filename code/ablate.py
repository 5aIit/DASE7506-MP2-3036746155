"""Validation ablations of the fixed model; no refitting or model selection."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from selector.training_data import load_training_split, metrics
from submission import load_checkpoint, learned_scores, apply_rules, raw_scores

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ablation.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new --output path to preserve the recorded ablation')
    with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=4, user_api='openmp'):
        bank, embeddings, _, _, answers, labels, hard = load_training_split('validation')
        model, config = load_checkpoint(ROOT / 'checkpoint')
        learned = learned_scores(bank, embeddings, model, config)
        names = [None, 'binding_head', 'format_transfer', 'target_units',
                 'fraction_repair', 'production_profit', 'inventory_profit']
        records = []
        for omit in names:
            base = learned_scores(bank, embeddings, model, config, True) if omit == 'binding_head' else learned.copy()
            scores = apply_rules(bank, base, config, omit)
            records.append({'name': 'full' if omit is None else 'without_' + omit,
                            **metrics(scores, answers, labels, hard)})
            if omit is None:
                np.testing.assert_array_equal(scores, raw_scores(bank, {'embeddings': embeddings}, ROOT / 'checkpoint'))
    result = {'protocol': 'one component removed at a time; all other settings unchanged; no refitting or retuning',
              'records': records, 'test_evaluated': False, 'validation_used_for_selection_previously': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
