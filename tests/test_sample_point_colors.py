import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.colmap_io.models import ImagePose, Point3D
from src.colmap_io.reconstructor import sample_point_colors


class TestSamplePointColors(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Two tiny synthetic images with known, distinct solid colors, so a sampled pixel's
        # value is unambiguous - image "a.png" is pure red, "b.png" is pure blue (BGR on disk,
        # since cv2.imwrite expects BGR and sample_point_colors converts back to RGB on load).
        red_bgr = np.full((10, 10, 3), (0, 0, 255), dtype=np.uint8)   # BGR pure red
        blue_bgr = np.full((10, 10, 3), (255, 0, 0), dtype=np.uint8)  # BGR pure blue
        cv2.imwrite(os.path.join(self.tmpdir, "a.png"), red_bgr)
        cv2.imwrite(os.path.join(self.tmpdir, "b.png"), blue_bgr)

        self.images = {
            1: ImagePose(
                image_id=1, name="a.png", qvec=np.zeros(4), tvec=np.zeros(3),
                R=np.eye(3), T=np.zeros((3, 1)), P=np.zeros((3, 4)), camera_id=1,
                points2d=[(5.0, 5.0, 100)],
            ),
            2: ImagePose(
                image_id=2, name="b.png", qvec=np.zeros(4), tvec=np.zeros(3),
                R=np.eye(3), T=np.zeros((3, 1)), P=np.zeros((3, 4)), camera_id=1,
                points2d=[(5.0, 5.0, 100)],
            ),
            3: ImagePose(
                image_id=3, name="missing.png", qvec=np.zeros(4), tvec=np.zeros(3),
                R=np.eye(3), T=np.zeros((3, 1)), P=np.zeros((3, 4)), camera_id=1,
                points2d=[(5.0, 5.0, 200)],
            ),
        }

    def test_single_observation_matches_real_pixel(self):
        pts3d = {100: Point3D(id=100, xyz=np.zeros(3), image_ids=[1], point2d_idxs=[0])}
        colors = sample_point_colors(pts3d, self.images, [self.tmpdir])
        np.testing.assert_array_equal(colors[100], [255, 0, 0])  # RGB pure red

    def test_averages_across_multiple_observations(self):
        pts3d = {100: Point3D(id=100, xyz=np.zeros(3), image_ids=[1, 2], point2d_idxs=[0, 0])}
        colors = sample_point_colors(pts3d, self.images, [self.tmpdir])
        np.testing.assert_array_equal(colors[100], [127, 0, 127])  # mean of red and blue

    def test_falls_back_to_gray_when_no_image_is_readable(self):
        pts3d = {200: Point3D(id=200, xyz=np.zeros(3), image_ids=[3], point2d_idxs=[0])}
        colors = sample_point_colors(pts3d, self.images, [self.tmpdir])
        np.testing.assert_array_equal(colors[200], [128, 128, 128])

    def test_out_of_bounds_pixel_is_skipped_not_crashed(self):
        images = dict(self.images)
        images[1] = ImagePose(
            image_id=1, name="a.png", qvec=np.zeros(4), tvec=np.zeros(3),
            R=np.eye(3), T=np.zeros((3, 1)), P=np.zeros((3, 4)), camera_id=1,
            points2d=[(500.0, 500.0, 100)],  # far outside the 10x10 image
        )
        pts3d = {100: Point3D(id=100, xyz=np.zeros(3), image_ids=[1], point2d_idxs=[0])}
        colors = sample_point_colors(pts3d, images, [self.tmpdir])
        np.testing.assert_array_equal(colors[100], [128, 128, 128])

    def test_returns_an_entry_for_every_point(self):
        pts3d = {
            100: Point3D(id=100, xyz=np.zeros(3), image_ids=[1], point2d_idxs=[0]),
            200: Point3D(id=200, xyz=np.zeros(3), image_ids=[3], point2d_idxs=[0]),
        }
        colors = sample_point_colors(pts3d, self.images, [self.tmpdir])
        self.assertEqual(set(colors.keys()), {100, 200})


if __name__ == "__main__":
    unittest.main()
