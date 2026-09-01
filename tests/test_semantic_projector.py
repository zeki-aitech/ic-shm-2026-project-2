import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reconstruction.semantic_projector import (
    SemanticProjector, vote_majority_class, CLASS_NAMES, CLASS_COLORS
)

class TestSemanticProjector(unittest.TestCase):
    def setUp(self):
        self.colmap_dir = "/workspaces/sfm_demo/data/Contest Dataset/camera_parameters"
        self.gt_masks_dir = "/workspaces/sfm_demo/outputs/gt_masks"
        self.output_dir = "/workspaces/sfm_demo/outputs/point_clouds"
        self.projector = SemanticProjector(self.colmap_dir, self.gt_masks_dir, self.output_dir)

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
        not os.path.exists("/workspaces/sfm_demo/outputs/gt_masks"),
        "Dataset outputs directory not mounted on current environment"
    )
    def test_full_projection_and_export(self):
        self.projector.preload_masks()
        self.assertGreaterEqual(len(self.projector.mask_cache), 300)

        classes, colors = self.projector.project()
        n_pts = len(classes)
        self.assertGreater(n_pts, 80000)
        self.assertEqual(len(colors), n_pts)

        # Verify class range
        for cid in classes.values():
            self.assertIn(cid, CLASS_NAMES)

        # Verify PLY export
        output_ply = self.projector.export_ply(
            os.path.join(self.output_dir, "test_semantic_bridge.ply")
        )
        self.assertTrue(os.path.exists(output_ply))
        self.assertGreater(os.path.getsize(output_ply), 100000)

        # Read first 15 lines of PLY file to verify header
        with open(output_ply, 'r', encoding='utf-8') as f:
            lines = [f.readline().strip() for _ in range(11)]

        self.assertEqual(lines[0], "ply")
        self.assertEqual(lines[1], "format ascii 1.0")
        self.assertTrue(lines[2].startswith("element vertex "))
        self.assertEqual(lines[-1], "end_header")


if __name__ == "__main__":
    unittest.main()
