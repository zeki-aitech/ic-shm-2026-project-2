import os
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.compose_splat_screenshots import compose_splat_screenshots


class TestComposeSplatScreenshots(unittest.TestCase):
    def test_writes_a_nonempty_png_wider_than_either_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            rgb_path = os.path.join(tmp, "rgb.png")
            sem_path = os.path.join(tmp, "sem.png")
            Image.new("RGB", (100, 60), "red").save(rgb_path)
            Image.new("RGB", (100, 60), "blue").save(sem_path)

            out_path = os.path.join(tmp, "fig6.png")
            returned = compose_splat_screenshots(rgb_path, sem_path, out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))

            combined = Image.open(out_path)
            self.assertGreater(combined.width, 200)  # 100 + 100 + gap
            self.assertGreater(combined.height, 60)  # + label height

    def test_handles_different_sized_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            rgb_path = os.path.join(tmp, "rgb.png")
            sem_path = os.path.join(tmp, "sem.png")
            Image.new("RGB", (200, 100), "red").save(rgb_path)
            Image.new("RGB", (80, 40), "blue").save(sem_path)

            out_path = os.path.join(tmp, "fig6.png")
            compose_splat_screenshots(rgb_path, sem_path, out_path)
            self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
