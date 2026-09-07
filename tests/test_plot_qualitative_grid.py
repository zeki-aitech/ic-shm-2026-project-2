import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.plot_qualitative_grid import build_qualitative_grid, colorize_mask


class TestColorizeMask(unittest.TestCase):
    def test_shape_and_dtype(self):
        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        rgb = colorize_mask(mask)
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_distinct_classes_get_distinct_colors(self):
        mask = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
        rgb = colorize_mask(mask)
        colors = {tuple(rgb[0, i]) for i in range(5)}
        self.assertEqual(len(colors), 5)


class TestBuildQualitativeGrid(unittest.TestCase):
    def _make_dummy_dataset(self, tmp, view_ids):
        render_dir = os.path.join(tmp, "renders")
        gt_images_dir = os.path.join(tmp, "images")
        gt_masks_dir = os.path.join(tmp, "masks")
        for d in (render_dir, gt_images_dir, gt_masks_dir):
            os.makedirs(d, exist_ok=True)

        rng = np.random.default_rng(0)
        for vid in view_ids:
            rgb = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
            mask = rng.integers(0, 5, (16, 16), dtype=np.uint8)
            Image.fromarray(rgb).save(os.path.join(render_dir, f"{vid}_rgb.png"))
            Image.fromarray(mask, mode="L").save(os.path.join(render_dir, f"{vid}_sem.png"))
            Image.fromarray(rgb).save(os.path.join(gt_images_dir, f"{vid}.png"))
            Image.fromarray(mask, mode="L").save(os.path.join(gt_masks_dir, f"{vid}.png"))
        return render_dir, gt_images_dir, gt_masks_dir

    def test_writes_a_nonempty_png_for_multiple_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_ids = ["001", "002", "003"]
            render_dir, gt_images_dir, gt_masks_dir = self._make_dummy_dataset(tmp, view_ids)
            out_path = os.path.join(tmp, "fig4.png")
            returned = build_qualitative_grid(view_ids, render_dir, gt_images_dir, gt_masks_dir, out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_single_view_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            view_ids = ["001"]
            render_dir, gt_images_dir, gt_masks_dir = self._make_dummy_dataset(tmp, view_ids)
            out_path = os.path.join(tmp, "fig4.png")
            build_qualitative_grid(view_ids, render_dir, gt_images_dir, gt_masks_dir, out_path)
            self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
