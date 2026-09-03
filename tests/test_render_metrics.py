import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.render_metrics import compute_psnr, compute_ssim

try:
    import torch
    import lpips  # noqa: F401

    _HAS_LPIPS = torch.cuda.is_available() or True  # lpips works on CPU too
except ImportError:
    _HAS_LPIPS = False


class TestPSNRSSIM(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.img_a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        self.img_b = self.img_a.copy()
        self.img_c = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

    def test_psnr_identical_is_very_high(self):
        self.assertGreater(compute_psnr(self.img_a, self.img_b), 60.0)

    def test_ssim_identical_is_one(self):
        self.assertAlmostEqual(compute_ssim(self.img_a, self.img_b), 1.0, places=5)

    def test_psnr_differs_for_different_images(self):
        p_same = compute_psnr(self.img_a, self.img_b)
        p_diff = compute_psnr(self.img_a, self.img_c)
        self.assertGreater(p_same, p_diff)

    def test_ssim_differs_for_different_images(self):
        s_same = compute_ssim(self.img_a, self.img_b)
        s_diff = compute_ssim(self.img_a, self.img_c)
        self.assertGreater(s_same, s_diff)


@unittest.skipUnless(_HAS_LPIPS, "lpips/torch not installed")
class TestLPIPS(unittest.TestCase):
    def test_lpips_identical_is_near_zero(self):
        from src.evaluation.render_metrics import compute_lpips

        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        self.assertLess(compute_lpips(img, img), 1e-3)


if __name__ == "__main__":
    unittest.main()
