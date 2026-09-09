import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.plot_ablation_convergence import plot_ablation_convergence

# class ids: 0=background, 1=deck, 2=stay_cable, 3=tower, 4=foundation
_EARLY = {0: 0.98, 1: 0.80, 2: 0.83, 3: 0.75, 4: 0.74}
_LATE = {0: 0.99, 1: 0.96, 2: 0.92, 3: 0.90, 4: 0.87}


class TestPlotAblationConvergence(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        curves = {
            "ours": {2000: _EARLY, 40000: _LATE},
            "no_warmstart": {2000: _EARLY, 40000: _LATE},
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "fig4.png")
            returned = plot_ablation_convergence(curves, out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_creates_missing_output_directory(self):
        curves = {"ours": {2000: _EARLY}}
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "nested", "dir", "fig4.png")
            plot_ablation_convergence(curves, nested)
            self.assertTrue(os.path.exists(nested))


if __name__ == "__main__":
    unittest.main()
