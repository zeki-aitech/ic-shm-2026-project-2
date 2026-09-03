import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gaussian_splatting.undistort import undistort_image, build_pinhole_K
from src.colmap_io.models import CameraIntrinsics


class TestUndistort(unittest.TestCase):
    def setUp(self):
        self.camera = CameraIntrinsics(
            id=1, model="SIMPLE_RADIAL", width=1320, height=989,
            f=925.70161892457077, cx=660.0, cy=494.5, k1=0.0089878633452682329,
        )
        img = np.zeros((989, 1320, 3), dtype=np.uint8)
        for y in range(0, 989, 50):
            img[y, :, :] = 255
        for x in range(0, 1320, 50):
            img[:, x, :] = 255
        self.img = img

    def test_dimensions_preserved(self):
        out = undistort_image(self.img, self.camera)
        self.assertEqual(out.shape, self.img.shape)

    def test_near_identity_at_center(self):
        out = undistort_image(self.img, self.camera)
        cy, cx = int(self.camera.cy), int(self.camera.cx)
        diff = np.abs(self.img[cy - 5 : cy + 5, cx - 5 : cx + 5].astype(int) - out[cy - 5 : cy + 5, cx - 5 : cx + 5].astype(int))
        self.assertLess(diff.mean(), 1.0)

    def test_measurable_shift_at_corner(self):
        out = undistort_image(self.img, self.camera)
        diff = np.abs(self.img[:20, :20].astype(int) - out[:20, :20].astype(int))
        self.assertGreater(diff.mean(), 5.0)

    def test_pinhole_K_shape(self):
        K = build_pinhole_K(self.camera)
        self.assertEqual(K.shape, (3, 3))
        self.assertAlmostEqual(K[0, 0], self.camera.f)
        self.assertAlmostEqual(K[0, 2], self.camera.cx)
        self.assertAlmostEqual(K[1, 2], self.camera.cy)


if __name__ == "__main__":
    unittest.main()
