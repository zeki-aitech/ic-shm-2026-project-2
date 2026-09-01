import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reconstruction.visualizer import read_ply_file, create_interactive_3d_figure


class TestVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.ply_path = os.path.join(cls.tmp_dir.name, "test_sample.ply")

        # Create a small valid ASCII PLY file
        n_points = 50
        with open(cls.ply_path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n_points}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("property int class_id\n")
            f.write("end_header\n")
            for i in range(n_points):
                f.write(f"{i*0.5:.2f} {i*0.2:.2f} {i*0.1:.2f} 255 0 0 1\n")

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_read_ply_file(self):
        xyz, rgb, cids = read_ply_file(self.ply_path)
        self.assertEqual(len(xyz), 50)
        self.assertEqual(len(rgb), len(xyz))
        self.assertEqual(len(cids), len(xyz))
        self.assertEqual(xyz.shape[1], 3)
        self.assertEqual(rgb.shape[1], 3)

    def test_create_interactive_3d_figure(self):
        xyz, rgb, cids = read_ply_file(self.ply_path)
        fig = create_interactive_3d_figure(xyz, rgb, cids, downsample_factor=1)
        self.assertIsNotNone(fig)
        self.assertGreaterEqual(len(fig.data), 1)


if __name__ == "__main__":
    unittest.main()
