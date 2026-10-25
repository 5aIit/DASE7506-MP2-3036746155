# Release 1.0 code

This directory contains the implemented selector, its training routine, and the
matching checkpoint. Start with [../REPRODUCE.md](../REPRODUCE.md) for exact commands.

- `submission.py`: the only prediction entry, implementing
  `predict(bank, arrays, checkpoint) -> list[int]`.
- `train.py`: refit the frozen recipe from supplied training data.
- `evaluate.py`: verify release hashes and run fresh official prediction/scoring.
- `ablate.py`: reproduce fixed-setting validation ablations.
- `audit.py`: verify integrity and run all synthetic regression tests.
- `selector/`: functional modules shared by training and inference.
- `checkpoint/`: one unchanged learned model and one consolidated configuration.
- `results/`: measurements and compact development evidence, never inference inputs.
- `release.json`: current frozen source/configuration/weight hashes.

The course files `common.py`, `predict.py`, `score_predictions.py`, `check_data.py`
and all `data/` files are unchanged. Their original interface documentation is
retained as [../guide/CODE_INTERFACE_ORIGINAL.md](../guide/CODE_INTERFACE_ORIGINAL.md).
The upstream package manifest under `guide/` describes the original starter,
not this release.

Official test accuracy: 395/467 = 84.58244111349036%.
Validation: 770/798; its selected hard development subset: 327/355.
Release 1.0 changes organization only; it is not a new model-selection experiment.
