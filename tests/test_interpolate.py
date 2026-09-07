import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gaussian_splatting.interpolate import interpolate_pose, slerp_quat


class TestSlerpQuat(unittest.TestCase):
    def test_t0_returns_first_quat(self):
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        result = slerp_quat(q1, q2, 0.0)
        np.testing.assert_allclose(result, q1, atol=1e-6)

    def test_t1_returns_second_quat(self):
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        result = slerp_quat(q1, q2, 1.0)
        np.testing.assert_allclose(result, q2, atol=1e-6)

    def test_result_is_unit_quaternion(self):
        q1 = np.array([0.7, 0.1, 0.2, 0.3])
        q2 = np.array([0.2, 0.6, -0.1, 0.4])
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            result = slerp_quat(q1, q2, t)
            self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)

    def test_takes_shorter_arc(self):
        # q2 is the "long way round" (negated) version of a quat close to q1; slerp should
        # still produce a short, smooth interpolation (norm stays 1, no sign-flip discontinuity).
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q_near = np.array([0.99, 0.01, 0.0, 0.0])
        q_near = q_near / np.linalg.norm(q_near)
        q2 = -q_near
        mid = slerp_quat(q1, q2, 0.5)
        # The midpoint should be close to q1/q_near, not orthogonal/far from both.
        self.assertGreater(abs(float(np.dot(mid, q1))), 0.9)


class TestInterpolatePose(unittest.TestCase):
    def test_endpoints_match_inputs(self):
        qvec1, tvec1 = (1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)
        qvec2, tvec2 = (0.0, 1.0, 0.0, 0.0), (4.0, 5.0, 6.0)

        R0, T0 = interpolate_pose(qvec1, tvec1, qvec2, tvec2, 0.0)
        R1, T1 = interpolate_pose(qvec1, tvec1, qvec2, tvec2, 1.0)

        np.testing.assert_allclose(T0, tvec1, atol=1e-6)
        np.testing.assert_allclose(T1, tvec2, atol=1e-6)
        self.assertEqual(R0.shape, (3, 3))
        self.assertEqual(R1.shape, (3, 3))

    def test_midpoint_translation_is_average(self):
        qvec1, tvec1 = (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        qvec2, tvec2 = (1.0, 0.0, 0.0, 0.0), (2.0, 4.0, 6.0)
        _, T = interpolate_pose(qvec1, tvec1, qvec2, tvec2, 0.5)
        np.testing.assert_allclose(T, [1.0, 2.0, 3.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
