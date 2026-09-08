import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.plot_training_curves import (
    parse_gaussian_log,
    parse_segmentation_log,
    plot_task_a_training_curve,
    plot_task_b_training_curve,
)

_SEG_LOG = """Some unrelated warning line
[segmentation] 300 labeled images -> train=240 val=60
[segmentation] epoch 1/80 loss=1.2991 val_mIoU=0.3191
[segmentation] epoch 2/80 loss=0.8537 val_mIoU=0.4086
[segmentation] epoch 3/80 loss=0.6609 val_mIoU=0.4517
"""

_GS_LOG = """[pycolmap] unrelated setup line
[gaussian_splatting] labeled=300 train=240 holdout=60
[gaussian_splatting] step 100/300 loss=0.3868 n_gaussians=84613 (2.3s)
[gaussian_splatting] step 200/300 loss=0.4850 n_gaussians=84613 (4.2s)
[gaussian_splatting] step 300/300 loss=0.3178 n_gaussians=90000 (7.9s)
"""


class TestParseSegmentationLog(unittest.TestCase):
    def test_extracts_epoch_loss_and_miou(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "seg.log")
            with open(log_path, "w") as f:
                f.write(_SEG_LOG)
            epochs, losses, mious = parse_segmentation_log(log_path)
            self.assertEqual(epochs, [1, 2, 3])
            self.assertEqual(losses, [1.2991, 0.8537, 0.6609])
            self.assertEqual(mious, [0.3191, 0.4086, 0.4517])


class TestParseGaussianLog(unittest.TestCase):
    def test_extracts_step_and_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "gs.log")
            with open(log_path, "w") as f:
                f.write(_GS_LOG)
            steps, losses = parse_gaussian_log(log_path)
            self.assertEqual(steps, [100, 200, 300])
            self.assertEqual(losses, [0.3868, 0.4850, 0.3178])


class TestPlotTaskATrainingCurve(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "seg.log")
            with open(log_path, "w") as f:
                f.write(_SEG_LOG)
            out_path = os.path.join(tmp, "fig7.png")
            returned = plot_task_a_training_curve(log_path, out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)


class TestPlotTaskBTrainingCurve(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "gs.log")
            with open(log_path, "w") as f:
                f.write(_GS_LOG)
            out_path = os.path.join(tmp, "fig8.png")
            returned = plot_task_b_training_curve(log_path, out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)


if __name__ == "__main__":
    unittest.main()
