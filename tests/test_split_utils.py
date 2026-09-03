import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import trajectory_interleaved_split


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


if __name__ == "__main__":
    unittest.main()
