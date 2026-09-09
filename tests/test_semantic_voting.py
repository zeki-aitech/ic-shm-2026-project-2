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

    def test_vote_majority_class_default_plain_plurality(self):
        # Default (strict_cable_majority=False): plain plurality, cable competing on equal
        # footing with every other class, ties broken by fixed priority (thin/rare structures,
        # cable included, favored over background/deck).
        self.assertEqual(vote_majority_class([1, 1, 0]), 1)
        self.assertEqual(vote_majority_class([3, 3, 2]), 3)
        self.assertEqual(vote_majority_class([2, 2, 2, 0]), 2)
        self.assertEqual(vote_majority_class([2, 2, 0]), 2)
        # Cable/background tie (2 each): cable wins the tie-break under plain plurality.
        self.assertEqual(vote_majority_class([2, 2, 0, 0]), 2)
        # Cable/deck tie (1 each): cable wins the tie-break.
        self.assertEqual(vote_majority_class([2, 1]), 2)
        self.assertEqual(vote_majority_class([1, 2]), 2)

        # Non-cable tie-break unchanged.
        self.assertEqual(vote_majority_class([1, 3]), 3)
        self.assertEqual(vote_majority_class([0, 1]), 1)

        # Empty fallback.
        self.assertEqual(vote_majority_class([]), 0)

    def test_vote_majority_class_strict_cable_majority_alternative(self):
        # strict_cable_majority=True: a tested-but-not-adopted alternative - cable is assigned
        # only with an absolute majority (>50%); otherwise cable votes are excluded and the
        # remaining classes compete by plurality.
        self.assertEqual(vote_majority_class([2, 2, 2, 0], strict_cable_majority=True), 2)
        self.assertEqual(vote_majority_class([2, 2, 0], strict_cable_majority=True), 2)
        self.assertNotEqual(vote_majority_class([2, 2, 0, 0], strict_cable_majority=True), 2)  # exactly 50% fails
        self.assertEqual(vote_majority_class([2, 1], strict_cable_majority=True), 1)  # no absolute majority for cable
        self.assertEqual(vote_majority_class([1, 2], strict_cable_majority=True), 1)
        # Non-cable cases are unaffected by the flag.
        self.assertEqual(vote_majority_class([1, 1, 0], strict_cable_majority=True), 1)
        self.assertEqual(vote_majority_class([], strict_cable_majority=True), 0)

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
