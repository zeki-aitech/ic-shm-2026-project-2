import os
import sys
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.colmap_io.semantic_voting import (
    SemanticProjector, vote_majority_class, CLASS_NAMES
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.getenv("CONTEST_DATASET_DIR", os.path.join(PROJECT_ROOT, "data", "Contest Dataset"))


class TestSemanticVoting(unittest.TestCase):
    def setUp(self):
        self.colmap_dir = os.path.join(DATASET_DIR, "camera_parameters")
        self.gt_masks_dir = os.path.join(PROJECT_ROOT, "outputs", "gt_masks")
        self.projector = SemanticProjector(self.colmap_dir, self.gt_masks_dir)

    def test_vote_majority_class(self):
        # Clear majority (non-cable)
        self.assertEqual(vote_majority_class([1, 1, 0]), 1)
        self.assertEqual(vote_majority_class([3, 3, 2]), 3)

        # stay_cable: absolute majority only
        self.assertEqual(vote_majority_class([2, 2, 2, 0]), 2)
        self.assertEqual(vote_majority_class([2, 2, 0]), 2)
        self.assertNotEqual(vote_majority_class([2, 2, 0, 0]), 2)  # exactly 50% fails
        self.assertEqual(vote_majority_class([2, 1]), 1)  # no absolute majority for cable

        # Cable cannot win via tie-break alone
        self.assertEqual(vote_majority_class([1, 2]), 1)

        # Non-cable tie-break unchanged
        self.assertEqual(vote_majority_class([1, 3]), 3)
        self.assertEqual(vote_majority_class([0, 1]), 1)

        # Empty fallback
        self.assertEqual(vote_majority_class([]), 0)

    @unittest.skipIf(
        not os.path.exists(os.path.join(PROJECT_ROOT, "outputs", "gt_masks")),
        "Dataset outputs directory not mounted on current environment"
    )
    def test_full_projection(self):
        self.projector.preload_masks()
        self.assertGreaterEqual(len(self.projector.mask_cache), 300)

        classes, colors = self.projector.project()
        n_pts = len(classes)
        self.assertGreater(n_pts, 80000)
        self.assertEqual(len(colors), n_pts)

        for cid in classes.values():
            self.assertIn(cid, CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
