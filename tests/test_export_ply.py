import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.colmap_io.models import Point3D
from src.gaussian_splatting.model import SemanticGaussianModel, NUM_CLASSES
from src.gaussian_splatting.export_ply import export_checkpoint_plys


@unittest.skipUnless(torch.cuda.is_available(), "gsplat requires a CUDA device")
class TestExportCheckpointPlys(unittest.TestCase):
    def test_writes_both_plys_from_a_saved_checkpoint(self):
        np.random.seed(0)
        n = 20
        pts3d = {i: Point3D(id=i, xyz=np.random.randn(3) * 0.5, image_ids=[], point2d_idxs=[]) for i in range(n)}
        point_classes = {i: int(np.random.randint(0, NUM_CLASSES)) for i in range(n)}
        point_colors = {i: np.random.randint(0, 255, 3) for i in range(n)}
        model = SemanticGaussianModel.init_from_sparse(pts3d, point_classes, point_colors, device="cuda")

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "final.pt")
            torch.save({"params": model.state_dict(), "step": 0}, ckpt_path)

            rgb_path, sem_path = export_checkpoint_plys(ckpt_path, tmp, "test", device="cuda")

            self.assertTrue(os.path.exists(rgb_path))
            self.assertTrue(os.path.exists(sem_path))
            for path in (rgb_path, sem_path):
                with open(path, "rb") as f:
                    self.assertEqual(f.read(3), b"ply")


if __name__ == "__main__":
    unittest.main()
