import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.colmap_io.models import CameraIntrinsics, ImagePose, Point3D


class TestModels(unittest.TestCase):
    def test_camera_intrinsics(self):
        cam = CameraIntrinsics(
            id=1,
            model="SIMPLE_RADIAL",
            width=1320,
            height=989,
            f=925.7016,
            cx=660.0,
            cy=494.5,
            k1=0.012
        )
        self.assertEqual(cam.id, 1)
        self.assertEqual(cam.model, "SIMPLE_RADIAL")
        self.assertEqual(cam.width, 1320)
        self.assertEqual(cam.height, 989)
        self.assertAlmostEqual(cam.f, 925.7016, places=4)
        self.assertAlmostEqual(cam.k1, 0.012, places=3)

    def test_image_pose(self):
        qvec = np.array([1.0, 0.0, 0.0, 0.0])
        tvec = np.array([0.0, 0.0, 0.0])
        R = np.eye(3)
        T = np.zeros((3, 1))
        P = np.hstack([R, T])

        pose = ImagePose(
            image_id=10,
            name="frame_001.png",
            qvec=qvec,
            tvec=tvec,
            R=R,
            T=T,
            P=P,
            camera_id=1,
            points2d=[(100.0, 200.0, 1)]
        )
        self.assertEqual(pose.image_id, 10)
        self.assertEqual(pose.name, "frame_001.png")
        self.assertEqual(pose.P.shape, (3, 4))
        self.assertEqual(len(pose.points2d), 1)

    def test_point3d(self):
        xyz = np.array([10.5, -2.3, 50.0])
        pt = Point3D(
            id=100,
            xyz=xyz,
            image_ids=[1, 2, 3],
            point2d_idxs=[0, 5, 2]
        )
        self.assertEqual(pt.id, 100)
        np.testing.assert_array_equal(pt.xyz, xyz)
        self.assertEqual(len(pt.image_ids), 3)
        self.assertEqual(len(pt.point2d_idxs), 3)


if __name__ == "__main__":
    unittest.main()
