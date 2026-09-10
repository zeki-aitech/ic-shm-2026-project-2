import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import trajectory_interleaved_split, train_val_test_split


class TestTrajectoryInterleavedSplit(unittest.TestCase):
    def test_holdout_size_matches_ratio(self):
        ids = list(range(300))
        train_ids, holdout_ids = trajectory_interleaved_split(ids, holdout_ratio=0.20)
        self.assertEqual(len(holdout_ids), 60)
        self.assertEqual(len(train_ids), 240)

    def test_disjoint_and_exhaustive(self):
        ids = [f"{i:03d}.png" for i in range(1, 301)]
        train_ids, holdout_ids = trajectory_interleaved_split(ids, holdout_ratio=0.20)
        self.assertEqual(set(train_ids) & set(holdout_ids), set())
        self.assertEqual(set(train_ids) | set(holdout_ids), set(ids))

    def test_every_fifth_is_holdout(self):
        ids = list(range(1, 301))
        _, holdout_ids = trajectory_interleaved_split(ids, holdout_ratio=0.20)
        self.assertEqual(holdout_ids, list(range(5, 301, 5)))

    def test_deterministic(self):
        ids = list(range(123))
        r1 = trajectory_interleaved_split(ids, holdout_ratio=0.2)
        r2 = trajectory_interleaved_split(ids, holdout_ratio=0.2)
        self.assertEqual(r1, r2)

    def test_train_order_preserved(self):
        ids = list(range(1, 21))
        train_ids, _ = trajectory_interleaved_split(ids, holdout_ratio=0.20)
        self.assertEqual(train_ids, sorted(train_ids))

    def test_zero_holdout_ratio(self):
        ids = list(range(10))
        train_ids, holdout_ids = trajectory_interleaved_split(ids, holdout_ratio=0.0)
        self.assertEqual(holdout_ids, [])
        self.assertEqual(train_ids, ids)


class TestTrainValTestSplit(unittest.TestCase):
    def test_sizes_match_80_10_10(self):
        ids = list(range(300))
        train_ids, val_ids, test_ids = train_val_test_split(ids, val_ratio=0.10, test_ratio=0.10)
        self.assertEqual(len(train_ids), 240)
        self.assertEqual(len(val_ids), 30)
        self.assertEqual(len(test_ids), 30)

    def test_disjoint_and_exhaustive(self):
        ids = [f"{i:03d}.png" for i in range(1, 301)]
        train_ids, val_ids, test_ids = train_val_test_split(ids, val_ratio=0.10, test_ratio=0.10)
        self.assertEqual(set(train_ids) & set(val_ids), set())
        self.assertEqual(set(train_ids) & set(test_ids), set())
        self.assertEqual(set(val_ids) & set(test_ids), set())
        self.assertEqual(set(train_ids) | set(val_ids) | set(test_ids), set(ids))

    def test_test_ids_match_plain_trajectory_split(self):
        """`test_ids` must be exactly what a single `trajectory_interleaved_split(ids,
        test_ratio)` call would hold out - Task B's `train.py`/`render_metrics.py` rely on this
        to keep the same never-touched test set Task A's own training excludes even from its
        internal validation."""
        ids = list(range(1, 301))
        _, direct_test_ids = trajectory_interleaved_split(ids, holdout_ratio=0.10)
        _, _, test_ids = train_val_test_split(ids, val_ratio=0.10, test_ratio=0.10)
        self.assertEqual(test_ids, direct_test_ids)

    def test_deterministic(self):
        ids = list(range(300))
        r1 = train_val_test_split(ids, 0.10, 0.10)
        r2 = train_val_test_split(ids, 0.10, 0.10)
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
