import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.colmap_io.models import Point3D
from src.gaussian_splatting.model import SemanticGaussianModel, NUM_CLASSES, _quat_to_rotmat
from src.gaussian_splatting.dataset import GSCamera
from src.gaussian_splatting.losses import photometric_loss, semantic_ce_loss


@unittest.skipUnless(torch.cuda.is_available(), "gsplat requires a CUDA device")
class TestGaussianRenderShapes(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        n = 30
        pts3d = {i: Point3D(id=i, xyz=np.random.randn(3) * 0.5, image_ids=[], point2d_idxs=[]) for i in range(n)}
        point_classes = {i: int(np.random.randint(0, NUM_CLASSES)) for i in range(n)}
        point_colors = {i: np.random.randint(0, 255, 3) for i in range(n)}
        self.model = SemanticGaussianModel.init_from_sparse(pts3d, point_classes, point_colors, device="cuda")
        self.camera = GSCamera(
            image_id=1, name="001.png",
            K=np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]]),
            R=np.eye(3), T=np.array([0.0, 0.0, 3.0]),
            width=64, height=48, image_path="", mask_path=None, is_holdout=False,
        )

    def test_output_shapes_and_ranges(self):
        rgb, sem = self.model.render(self.camera)
        self.assertEqual(tuple(rgb.shape), (48, 64, 3))
        self.assertEqual(tuple(sem.shape), (48, 64, NUM_CLASSES))
        self.assertGreaterEqual(rgb.min().item(), 0.0)
        self.assertLessEqual(rgb.max().item(), 1.0)
        self.assertFalse(torch.isnan(rgb).any().item())
        self.assertFalse(torch.isnan(sem).any().item())

    def test_backward_pass(self):
        rgb, sem = self.model.render(self.camera)
        target_rgb = torch.rand_like(rgb)
        target_mask = torch.randint(0, NUM_CLASSES, (48, 64), device=rgb.device)
        loss = photometric_loss(rgb, target_rgb) + semantic_ce_loss(sem, target_mask)
        loss.backward()
        self.assertIsNotNone(self.model.means.grad)
        self.assertGreater(self.model.means.grad.norm().item(), 0.0)
        self.assertIsNotNone(self.model.sem_logits.grad)

    def test_identity_pose_delta_matches_uncorrected_render(self):
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
        zero_trans = torch.zeros(3, device="cuda")
        out_a, _, _ = self.model.render_full(self.camera, packed=True)
        out_b, _, _ = self.model.render_full(
            self.camera, packed=True, pose_delta_quat=identity_quat, pose_delta_trans=zero_trans
        )
        torch.testing.assert_close(out_a, out_b)

    def test_pose_delta_gradients_flow(self):
        delta_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda", requires_grad=True)
        delta_trans = torch.zeros(3, device="cuda", requires_grad=True)
        out, _alpha, _meta = self.model.render_full(
            self.camera, packed=True, pose_delta_quat=delta_quat, pose_delta_trans=delta_trans
        )
        rgb = out[0][..., :3].clamp(0, 1)
        loss = photometric_loss(rgb, torch.rand_like(rgb))
        loss.backward()
        self.assertIsNotNone(delta_quat.grad)
        self.assertIsNotNone(delta_trans.grad)
        self.assertIsNotNone(self.model.means.grad)


class TestQuatToRotmat(unittest.TestCase):
    def test_identity_quaternion(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        R = _quat_to_rotmat(q)
        torch.testing.assert_close(R, torch.eye(3))


if __name__ == "__main__":
    unittest.main()
