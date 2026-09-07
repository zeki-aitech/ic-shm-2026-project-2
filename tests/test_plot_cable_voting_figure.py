import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.plot_cable_voting_figure import (
    build_cable_voting_figure,
    draw_voting_schematic,
    render_overlay_crop,
)


class TestRenderOverlayCrop(unittest.TestCase):
    def test_shape_matches_crop_box(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "img.png")
            mask_path = os.path.join(tmp, "mask.png")
            Image.fromarray(rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)).save(img_path)
            Image.fromarray(rng.integers(0, 5, (20, 30), dtype=np.uint8), mode="L").save(mask_path)

            overlay = render_overlay_crop(img_path, mask_path, (2, 10, 3, 15))
            self.assertEqual(overlay.shape, (8, 12, 3))
            self.assertEqual(overlay.dtype, np.uint8)

    def test_background_pixels_untinted(self):
        img = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)  # all background
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "img.png")
            mask_path = os.path.join(tmp, "mask.png")
            Image.fromarray(img).save(img_path)
            Image.fromarray(mask, mode="L").save(mask_path)

            overlay = render_overlay_crop(img_path, mask_path, (0, 4, 0, 4))
            np.testing.assert_array_equal(overlay, img)


class TestDrawVotingSchematic(unittest.TestCase):
    def test_does_not_crash(self):
        fig, ax = plt.subplots()
        draw_voting_schematic(ax)
        plt.close(fig)


class TestBuildCableVotingFigure(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "img.png")
            mask_path = os.path.join(tmp, "mask.png")
            Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(img_path)
            Image.fromarray(rng.integers(0, 5, (40, 40), dtype=np.uint8), mode="L").save(mask_path)

            out_path = os.path.join(tmp, "fig2.png")
            returned = build_cable_voting_figure(img_path, mask_path, (0, 40, 0, 40), out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)


if __name__ == "__main__":
    unittest.main()
