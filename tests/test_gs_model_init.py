import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.colmap_io.models import Point3D
from src.gaussian_splatting.model import SemanticGaussianModel, NUM_CLASSES


@unittest.skipUnless(torch.cuda.is_available(), "gsplat requires a CUDA device")
class TestGaussianModelInit(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.n = 50
        self.pts3d = {
            i: Point3D(id=i, xyz=np.random.randn(3) * 0.5, image_ids=[], point2d_idxs=[])
            for i in range(self.n)
        }
        self.point_classes = {i: int(np.random.randint(0, NUM_CLASSES)) for i in range(self.n)}
        self.point_colors = {i: np.random.randint(0, 255, 3) for i in range(self.n)}
        self.model = SemanticGaussianModel.init_from_sparse(
            self.pts3d, self.point_classes, self.point_colors, device="cuda"
        )

    def test_shapes(self):
        self.assertEqual(self.model.means.shape, (self.n, 3))
        self.assertEqual(self.model.log_scales.shape, (self.n, 3))
        self.assertEqual(self.model.quats.shape, (self.n, 4))
        self.assertEqual(self.model.opacity_logits.shape, (self.n,))
        self.assertEqual(self.model.color_logits.shape, (self.n, 3))
        self.assertEqual(self.model.sem_logits.shape, (self.n, NUM_CLASSES))

    def test_no_nans(self):
        for p in self.model.parameters():
            self.assertFalse(torch.isnan(p).any().item())

    def test_semantic_warm_start_matches_voted_class(self):
        argmax = self.model.sem_logits.detach().cpu().numpy().argmax(axis=1)
        for i in range(self.n):
            self.assertEqual(argmax[i], self.point_classes[i])

    def test_means_match_input_xyz(self):
        means = self.model.means.detach().cpu().numpy()
        expected = np.stack([self.pts3d[i].xyz for i in sorted(self.pts3d.keys())])
        np.testing.assert_allclose(means, expected, atol=1e-4)

    def test_export_ply(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.model.export_ply(os.path.join(tmp, "cloud.ply"))
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                header = [next(f) for _ in range(3)]
            self.assertEqual(header[0].strip(), "ply")
            self.assertIn(f"element vertex {self.n}", header[2])

    def test_export_splat_ply(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.model.export_splat_ply(os.path.join(tmp, "splat.ply"))
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as f:
                magic = f.read(3)
            self.assertEqual(magic, b"ply")

    def test_export_semantic_splat_ply(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.model.export_semantic_splat_ply(os.path.join(tmp, "splat_sem.ply"))
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as f:
                magic = f.read(3)
            self.assertEqual(magic, b"ply")


if __name__ == "__main__":
    unittest.main()
