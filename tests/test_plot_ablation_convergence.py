import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.plot_ablation_convergence import plot_ablation_convergence

STAY_CABLE_CLASS_ID = 2


class TestPlotAblationConvergence(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        curves = {
            "strict_majority": {2000: {STAY_CABLE_CLASS_ID: 0.833}, 40000: {STAY_CABLE_CLASS_ID: 0.9213}},
            "plain_plurality": {2000: {STAY_CABLE_CLASS_ID: 0.8265}, 40000: {STAY_CABLE_CLASS_ID: 0.9236}},
            "no_warmstart": {2000: {STAY_CABLE_CLASS_ID: 0.8098}, 40000: {STAY_CABLE_CLASS_ID: 0.9202}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "fig9.png")
            returned = plot_ablation_convergence(curves, STAY_CABLE_CLASS_ID, out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_creates_missing_output_directory(self):
        curves = {"baseline": {2000: {STAY_CABLE_CLASS_ID: 0.833}}}
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "nested", "dir", "fig9.png")
            plot_ablation_convergence(curves, STAY_CABLE_CLASS_ID, nested)
            self.assertTrue(os.path.exists(nested))


if __name__ == "__main__":
    unittest.main()
