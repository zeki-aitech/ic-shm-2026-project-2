import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reconstruction.colmap_parser import ColmapParser, CameraIntrinsics, ImagePose, Point3D

class TestColmapParser(unittest.TestCase):
    def setUp(self):
        self.colmap_dir = "/workspaces/sfm_demo/data/Contest Dataset/camera_parameters"
        self.parser = ColmapParser(self.colmap_dir)

    def test_parse_cameras(self):
        cam = self.parser.parse_cameras()
        self.assertIsInstance(cam, CameraIntrinsics)
        self.assertEqual(cam.id, 1)
        self.assertEqual(cam.width, 1320)
        self.assertEqual(cam.height, 989)
        self.assertAlmostEqual(cam.f, 925.7016, places=3)
        self.assertAlmostEqual(cam.cx, 660.0, places=1)
        self.assertAlmostEqual(cam.cy, 494.5, places=1)

    def test_parse_images(self):
        self.parser.parse_cameras()
        images = self.parser.parse_images()
        self.assertEqual(len(images), 400)
        
        first_img = next(iter(images.values()))
        self.assertIsInstance(first_img, ImagePose)
        self.assertEqual(first_img.P.shape, (3, 4))
        self.assertGreater(len(first_img.points2d), 0)

    def test_triangulation(self):
        cam, images, points3d = self.parser.load()
        self.assertGreater(len(points3d), 80000)
        self.assertLess(len(points3d), 86336)  # outlier removal should reduce count
        
        first_pt = next(iter(points3d.values()))
        self.assertIsInstance(first_pt, Point3D)
        self.assertEqual(len(first_pt.xyz), 3)
        self.assertFalse(np.isnan(first_pt.xyz).any())
        self.assertGreaterEqual(len(first_pt.image_ids), 2)

if __name__ == "__main__":
    unittest.main()
