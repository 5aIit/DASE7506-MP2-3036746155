"""Release contracts on synthetic questions; scoring references are never read.

Run from code/: python -m unittest discover -s tests -v
MP2_CHECKPOINT may point to a separately reproduced trusted checkpoint.
"""
import builtins
from copy import deepcopy
import io
from itertools import permutations
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from common import parse_answer, validate_choices
from selector.features import answer_groups
from selector.answers import aggregate_scores, choose
import submission



def synthetic_inputs():
    """Unseen rows include equivalent answers and missing answer markers."""
    texts = [
        ["Half of one is 0.5. #### 1/2", "One divided by two. #### 0.5",
         "Add one and two. #### 3", "There is insufficient information."],
        ["Three bags with two each: 3 * 2 = 6. #### 6",
         "The total is six. \\boxed{6}", "Three plus two is five. #### 5",
         "Three minus two is one. #### 1"],
        ["Answer missing", "No final numeric answer",
         "Reasoning only: 7 times 9 is 63.", "\\boxed{unknown}"],
        ["This explanation has no final marker", "Two dozen is twenty-four. #### 24",
         "An incomplete reasoning trace", "#### invalid"],
    ]
    bank = [
        {"id": f"synthetic-{i}", "question": question,
         "candidates": [{"index": j, "text": text}
                        for j, text in enumerate(candidates)]}
        for i, (question, candidates) in enumerate(zip(
            ["What is half of one?", "How many objects are in three bags of two?",
             "What does an answerless group do?", "How many objects are two dozen?"],
            texts,
        ))
    ]
    rng = np.random.default_rng(7506)
    arrays = {"embeddings": rng.normal(size=(len(bank), 4, 1024)).astype(np.float32)}
    return bank, arrays

def is_reference_path(path):
    """Allow checkpoint reads while rejecting all split reference JSONL files."""
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False  # File descriptors can be used by normal library code.
    parsed = Path(os.fsdecode(path))
    return (parsed.name in {f"gsm8k_{split}.jsonl"
                            for split in ("train", "validation", "test", "audit")}
            and parsed.parent.name != "candidates")

class AnswerHandlingTests(unittest.TestCase):
    def test_normalized_answers_are_grouped_without_duplicate_weight(self):
        bank, _ = synthetic_inputs()
        answers = answer_groups(bank)
        self.assertEqual(answers[0], ["1/2", "1/2", "3", None])
        self.assertEqual(answers[1], ["6", "6", "5", "1"])
        score = np.array([[2., 6., 5., 100.]])
        grouped = aggregate_scores(score, answers[:1], "answer_mean")
        np.testing.assert_array_equal(grouped, [[4., 4., 5., 100.]])
        self.assertEqual(choose(grouped, answers[:1]), [2])

    def test_unparseable_candidates_are_masked_when_possible(self):
        self.assertEqual(choose(np.array([[100., 2., 80., 70.]]),
                                [[None, "24", None, None]]), [1])
        result = choose(np.zeros((1, 4)), [[None] * 4])
        self.assertEqual(len(result), 1)
        self.assertIs(type(result[0]), int)
        self.assertIn(result[0], range(4))

class SubmissionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = Path(os.environ.get("MP2_CHECKPOINT", CODE / "checkpoint"))
        if not (cls.checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Missing release checkpoint: {cls.checkpoint}")

    def setUp(self):
        self.bank, self.arrays = synthetic_inputs()

    def run_predict(self, bank=None, arrays=None):
        bank = self.bank if bank is None else bank
        arrays = self.arrays if arrays is None else arrays
        result = submission.predict(bank, arrays, self.checkpoint)
        validate_choices(bank, [row["id"] for row in bank], result)
        return result

    def test_single_batch_and_empty_inputs(self):
        self.assertEqual(len(self.run_predict()), len(self.bank))
        self.assertEqual(len(self.run_predict(self.bank[:1],
            {"embeddings": self.arrays["embeddings"][:1]})), 1)
        self.assertEqual(self.run_predict([], {"embeddings": np.empty((0, 4, 1024),
                                                                     dtype=np.float32)}), [])

    def test_reordering_questions_only_reorders_predictions(self):
        baseline = self.run_predict()
        order = [3, 0, 2, 1]
        reordered = self.run_predict([self.bank[i] for i in order],
                                     {"embeddings": self.arrays["embeddings"][order]})
        self.assertEqual(reordered, [baseline[i] for i in order])

    def test_ids_and_unrelated_candidate_metadata_are_ignored(self):
        baseline = self.run_predict()
        altered = deepcopy(self.bank)
        for i, row in enumerate(altered):
            row["id"] = f"unseen-audit-id-{100_000 - i}"
            row["code"] = "unused arbitrary question metadata"
            for j, candidate in enumerate(row["candidates"]):
                candidate["code"] = {"arbitrary": f"metadata-{i}-{j}"}
        self.assertEqual(self.run_predict(altered), baseline)

    def test_batching_and_duplicate_rows_do_not_change_predictions(self):
        baseline = self.run_predict()
        individual = [self.run_predict([row], {"embeddings": self.arrays["embeddings"][i:i+1]})[0]
                      for i, row in enumerate(self.bank)]
        self.assertEqual(individual, baseline)
        duplicates = [deepcopy(self.bank[0]), deepcopy(self.bank[0])]
        duplicates[1]["id"] = "new-id-same-question"
        repeated = np.repeat(self.arrays["embeddings"][:1], 2, axis=0)
        self.assertEqual(self.run_predict(duplicates, {"embeddings": repeated}), [baseline[0]] * 2)

    def test_predict_never_opens_references(self):
        real_builtin_open, real_io_open = builtins.open, io.open

        def guarded(real_open):
            def open_without_references(file, *args, **kwargs):
                if is_reference_path(file):
                    raise AssertionError(f"Prediction attempted reference IO: {file}")
                return real_open(file, *args, **kwargs)
            return open_without_references

        with patch("builtins.open", guarded(real_builtin_open)), patch("io.open", guarded(real_io_open)):
            self.run_predict()

    def test_inputs_are_unchanged_and_repeated_calls_are_deterministic(self):
        original_bank = deepcopy(self.bank)
        original_arrays = {key: value.copy() for key, value in self.arrays.items()}
        first = self.run_predict()
        second = self.run_predict()
        self.assertEqual(first, second)
        self.assertEqual(self.bank, original_bank)
        self.assertEqual(set(self.arrays), set(original_arrays))
        for key in original_arrays:
            np.testing.assert_array_equal(self.arrays[key], original_arrays[key])

    def test_only_parseable_candidate_is_selected(self):
        self.assertEqual(self.run_predict()[3], 1)

    def test_uncompressed_bundle_is_within_asset_limit(self):
        # train.py writes joblib assets with compress=0, so stored size is applicable.
        asset_bytes = sum(path.stat().st_size for path in self.checkpoint.rglob("*") if path.is_file())
        self.assertGreater(asset_bytes, 0)
        self.assertLessEqual(asset_bytes, 16 * 2**20)

    def test_invalid_embedding_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_predict(arrays={'embeddings': np.zeros((4, 4, 12))})

    def test_nonfinite_embeddings_are_rejected(self):
        self.arrays['embeddings'][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.run_predict()

    def test_candidate_permutation_preserves_selected_numeric_answer(self):
        baseline = self.run_predict()
        order = [2, 0, 3, 1]
        bank = deepcopy(self.bank)
        for row in bank:
            row['candidates'] = [row['candidates'][j] for j in order]
            for j, candidate in enumerate(row['candidates']):
                candidate['index'] = j
        selected = self.run_predict(bank, {'embeddings': self.arrays['embeddings'][:, order]})
        for i, (old, new) in enumerate(zip(baseline, selected)):
            self.assertEqual(parse_answer(self.bank[i]['candidates'][old]['text']),
                             parse_answer(bank[i]['candidates'][new]['text']))

    def test_all_candidate_permutations_preserve_scores(self):
        with threadpool_limits(limits=1,user_api='blas'),threadpool_limits(limits=4,user_api='openmp'):
            base=submission.raw_scores(self.bank,self.arrays,self.checkpoint)
            for order in permutations(range(4)):
                with self.subTest(order=order):
                    bank=deepcopy(self.bank)
                    for row in bank:
                        row['candidates']=[row['candidates'][j] for j in order]
                        for j,candidate in enumerate(row['candidates']):candidate['index']=j
                    scores=submission.raw_scores(bank,{'embeddings':self.arrays['embeddings'][:,list(order)]},self.checkpoint)
                    np.testing.assert_allclose(scores,base[:,list(order)],rtol=1e-7,atol=1e-7)

    def test_zero_embeddings_and_infinities(self):
        values={'embeddings':np.zeros_like(self.arrays['embeddings'])}
        self.run_predict(arrays=values)
        for v in (np.inf,-np.inf):
            values['embeddings'][0,0,0]=v
            with self.assertRaises(ValueError):self.run_predict(arrays=values)

    def test_wrong_candidate_count(self):
        bank=deepcopy(self.bank);bank[0]['candidates'].pop()
        with self.assertRaises(ValueError):self.run_predict(bank=bank)


if __name__ == "__main__":
    unittest.main()
