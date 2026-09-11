import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.plot_per_class_iou import plot_per_class_iou, read_report_ious


class TestPlotPerClassIoU(unittest.TestCase):
    def test_reads_class_ious_without_treating_aggregate_miou_as_a_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.md")
            with open(path, "w") as f:
                f.write("- mIoU: 0.5\n  - deck: 0.9\n  - stay_cable: 0.8\n"
                        "  - tower: 0.7\n  - foundation: 0.6\n  - background: 0.99\n")
            self.assertEqual(read_report_ious(path), {0: 0.99, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6})

    def test_writes_a_nonempty_png(self):
        iou_per_class = {0: 0.99, 1: 0.95, 2: 0.92, 3: 0.91, 4: 0.87}
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "fig.png")
            returned = plot_per_class_iou(iou_per_class, out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_background_excluded_does_not_crash_structural_miou(self):
        # Only structural classes present (no background key) - should still compute fine.
        iou_per_class = {1: 0.95, 2: 0.92, 3: 0.91, 4: 0.87}
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "fig.png")
            plot_per_class_iou(iou_per_class, out_path)
            self.assertTrue(os.path.exists(out_path))

    def test_explicit_structural_miou_used_over_computed(self):
        # Passing an explicit value should not raise even if it doesn't match the mean.
        iou_per_class = {0: 0.99, 1: 0.95, 2: 0.92, 3: 0.91, 4: 0.87}
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "fig.png")
            plot_per_class_iou(iou_per_class, out_path, structural_miou=0.9147)
            self.assertTrue(os.path.exists(out_path))

    def test_creates_missing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "nested", "dir", "fig.png")
            plot_per_class_iou({1: 0.95, 2: 0.92, 3: 0.91, 4: 0.87}, nested)
            self.assertTrue(os.path.exists(nested))


if __name__ == "__main__":
    unittest.main()
