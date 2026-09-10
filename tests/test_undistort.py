import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gaussian_splatting.undistort import undistort_image, undistort_mask, build_pinhole_K
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

        # A synthetic 5-class mask: four big interior quadrants (classes 1-4) leaving a
        # background (0) border - like a real mask, where 0 is always present at the frame
        # edges, so `cv2.remap`'s zero-fill for out-of-bounds source pixels never introduces a
        # class id that wasn't already in the mask.
        mask = np.zeros((989, 1320), dtype=np.uint8)
        mask[50:495, 50:660] = 1
        mask[50:495, 660:1270] = 2
        mask[495:939, 50:660] = 3
        mask[495:939, 660:1270] = 4
        self.mask = mask

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

    def test_mask_dimensions_and_dtype_preserved(self):
        out = undistort_mask(self.mask, self.camera)
        self.assertEqual(out.shape, self.mask.shape)
        self.assertEqual(out.dtype, self.mask.dtype)

    def test_mask_stays_discrete_no_new_classes(self):
        """Nearest-neighbor interpolation must never blend two class ids into a value that was
        never in the input - the failure mode `cv2.undistort`'s default (bilinear/cubic)
        interpolation would cause if used on a label map."""
        out = undistort_mask(self.mask, self.camera)
        self.assertTrue(set(np.unique(out)) <= set(np.unique(self.mask)))

    def test_mask_near_identity_at_center(self):
        out = undistort_mask(self.mask, self.camera)
        cy, cx = int(self.camera.cy), int(self.camera.cx)
        # A window straddling all four quadrants right at the (near-zero-distortion) principal
        # point should be exactly unchanged.
        np.testing.assert_array_equal(
            self.mask[cy - 5 : cy + 5, cx - 5 : cx + 5], out[cy - 5 : cy + 5, cx - 5 : cx + 5]
        )

    def test_mask_measurable_shift_at_corner(self):
        out = undistort_mask(self.mask, self.camera)
        # A window straddling the background/class-1 boundary near the top-left corner (where
        # distortion is largest) should shift measurably, unlike the untouched interior.
        window = np.s_[40:60, 40:60]
        self.assertFalse(np.array_equal(self.mask[window], out[window]))


if __name__ == "__main__":
    unittest.main()
