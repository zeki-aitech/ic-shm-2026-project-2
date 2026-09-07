import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.plot_interpolation_sequence import _discover_frames, build_interpolation_filmstrip


def _write_dummy_frames(frame_dir, specs):
    os.makedirs(frame_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for idx, t in specs:
        stem = f"t{idx:02d}_{t:.2f}"
        rgb = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        sem = rng.integers(0, 5, (8, 8), dtype=np.uint8)
        Image.fromarray(rgb).save(os.path.join(frame_dir, f"{stem}_rgb.png"))
        Image.fromarray(sem, mode="L").save(os.path.join(frame_dir, f"{stem}_sem.png"))


class TestDiscoverFrames(unittest.TestCase):
    def test_sorted_by_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dummy_frames(tmp, [(2, 1.0), (0, 0.0), (1, 0.5)])
            frames = _discover_frames(tmp)
            self.assertEqual([f[0] for f in frames], [0, 1, 2])
            self.assertEqual([f[1] for f in frames], [0.0, 0.5, 1.0])

    def test_ignores_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_dummy_frames(tmp, [(0, 0.0)])
            with open(os.path.join(tmp, "notes.txt"), "w") as f:
                f.write("hello")
            frames = _discover_frames(tmp)
            self.assertEqual(len(frames), 1)


class TestBuildInterpolationFilmstrip(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_dir = os.path.join(tmp, "frames")
            _write_dummy_frames(frame_dir, [(0, 0.0), (1, 0.5), (2, 1.0)])
            out_path = os.path.join(tmp, "fig5.png")
            returned = build_interpolation_filmstrip(frame_dir, out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_raises_on_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_interpolation_filmstrip(tmp, os.path.join(tmp, "out.png"))


if __name__ == "__main__":
    unittest.main()
