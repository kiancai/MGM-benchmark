from __future__ import annotations

import hashlib
import sys
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MGM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MGM_ROOT / "scripts"))

from run_task_mgm5db import (  # noqa: E402
    EXPECTED_N_GENERA,
    MAX_LEN,
    _write_split_corpus,
    align_case_scores,
    legacy_inner_split,
)
from verify_task_mgm5db import artifact_tree_fingerprint  # noqa: E402


class DummyTokenizer:
    pad_token_id = 0
    bos_token_id = 2
    eos_token_id = 3


class LegacyInnerSplitTests(unittest.TestCase):
    def test_is_deterministic_and_covers_outer_train(self):
        indices = np.arange(100, 120, dtype=np.int64)
        labels = np.asarray([0, 1] * 10, dtype=np.int64)
        first = legacy_inner_split(indices, labels, 0.15, 0)
        second = legacy_inner_split(indices, labels, 0.15, 0)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        train_idx, val_idx, train_y, val_y = first
        self.assertEqual(len(val_idx), 3)
        self.assertEqual(set(np.concatenate([train_idx, val_idx])), set(indices))
        self.assertEqual(set(train_y), {0, 1})
        self.assertEqual(set(val_y), {0, 1})

    def test_rejects_one_class_validation(self):
        indices = np.arange(6, dtype=np.int64)
        labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "validation"):
            legacy_inner_split(indices, labels, 0.15, 0)


class ScoreAlignmentTests(unittest.TestCase):
    def _write(self, frame: pd.DataFrame, directory: str) -> Path:
        path = Path(directory) / "y_score.csv"
        frame.to_csv(path)
        return path

    def test_reorders_rows_and_returns_case_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.DataFrame(
                {"case": [2.0, -1.0], "control": [0.0, 1.0]},
                index=["sample_b", "sample_a"],
            )
            scores = align_case_scores(
                self._write(frame, directory), ["sample_a", "sample_b"]
            )
            expected = np.asarray([
                np.exp(-1.0) / (np.exp(1.0) + np.exp(-1.0)),
                np.exp(2.0) / (np.exp(0.0) + np.exp(2.0)),
            ])
            np.testing.assert_allclose(scores, expected)

    def test_rejects_missing_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.DataFrame(
                {"control": [0.0], "case": [1.0]}, index=["sample_a"]
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                align_case_scores(
                    self._write(frame, directory), ["sample_a", "sample_b"]
                )


class TokenizationContractTests(unittest.TestCase):
    def test_truncated_sequence_matches_mgm5db_streaming_builder(self):
        counts = np.zeros((1, EXPECTED_N_GENERA), dtype=np.float64)
        counts[0, :600] = np.arange(1, 601, dtype=np.float64)
        sample_ids = np.asarray(["sample_a"])
        labels = np.asarray([1], dtype=np.int64)
        genus_token_ids = np.arange(4, 4 + EXPECTED_N_GENERA, dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.pkl"
            labels_path = Path(directory) / "labels.csv"
            stats = _write_split_corpus(
                counts,
                sample_ids,
                labels,
                DummyTokenizer(),
                np.zeros(EXPECTED_N_GENERA, dtype=np.float64),
                np.ones(EXPECTED_N_GENERA, dtype=np.float64),
                genus_token_ids,
                corpus_path,
                labels_path,
            )
            with corpus_path.open("rb") as handle:
                corpus = pickle.load(handle)
        tokens = corpus.tokens.numpy()[0]
        # Actual MGM-5DB builder uses n=min(len(sent)+2, 512), then
        # BOS + sent[:n-2] + EOS.  A truncated row therefore keeps EOS.
        self.assertEqual(tokens.shape, (MAX_LEN,))
        self.assertEqual(tokens[0], DummyTokenizer.bos_token_id)
        self.assertEqual(tokens[-1], DummyTokenizer.eos_token_id)
        np.testing.assert_array_equal(
            tokens[1:-1],
            genus_token_ids[np.arange(599, 89, -1)],
        )
        self.assertEqual(stats["n_truncated"], 1)


class ArtifactTreeFingerprintTests(unittest.TestCase):
    def test_is_relative_order_stable_and_content_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z").mkdir()
            first = root / "z/result.csv"
            second = root / "a.json"
            first.write_text("result\n", encoding="utf-8")
            second.write_text("audit\n", encoding="utf-8")

            observed = artifact_tree_fingerprint([first, second], root)
            reversed_order = artifact_tree_fingerprint([second, first], root)
            self.assertEqual(observed, reversed_order)
            self.assertEqual(observed["n_files"], 2)

            digest = hashlib.sha256()
            for path in sorted([first, second], key=lambda item: item.relative_to(root).as_posix()):
                digest.update(path.relative_to(root).as_posix().encode("utf-8"))
                digest.update(b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
                digest.update(b"\n")
            self.assertEqual(observed["tree_sha256"], digest.hexdigest())

            first.write_text("changed\n", encoding="utf-8")
            self.assertNotEqual(
                observed,
                artifact_tree_fingerprint([first, second], root),
            )


if __name__ == "__main__":
    unittest.main()
