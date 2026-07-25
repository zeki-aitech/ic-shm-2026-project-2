import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reconstruction.visualizer import read_ply_file, create_interactive_3d_figure

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.ply_path = "/workspaces/sfm_demo/outputs/point_clouds/semantic_bridge_sparse.ply"

    def test_read_ply_file(self):
        xyz, rgb, cids = read_ply_file(self.ply_path)
        self.assertGreater(len(xyz), 80000)
        self.assertEqual(len(rgb), len(xyz))
        self.assertEqual(len(cids), len(xyz))
        self.assertEqual(xyz.shape[1], 3)
        self.assertEqual(rgb.shape[1], 3)

    def test_create_interactive_3d_figure(self):
        xyz, rgb, cids = read_ply_file(self.ply_path)
        fig = create_interactive_3d_figure(xyz, rgb, cids, downsample_factor=10)
        self.assertIsNotNone(fig)
        self.assertGreaterEqual(len(fig.data), 1)

if __name__ == "__main__":
    unittest.main()
