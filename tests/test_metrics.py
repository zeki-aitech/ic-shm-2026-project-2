"""Unit tests for src/evaluation/metrics.py."""
import unittest
import numpy as np
from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_iou_per_class,
    compute_miou,
    compute_overall_accuracy,
    compute_class_accuracy,
    compute_f1_scores,
)


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        # Perfect match case
        self.y_true_perfect = np.array([0, 1, 2, 3, 4, 1, 2, 3])
        self.y_pred_perfect = np.array([0, 1, 2, 3, 4, 1, 2, 3])

        # Imbalanced and imperfect case
        self.y_true = np.array([1, 1, 1, 1, 2, 2, 3, 3, 4, 0])
        self.y_pred = np.array([1, 1, 1, 2, 2, 2, 3, 0, 4, 0])

    def test_confusion_matrix_shape_and_values(self):
        cm = compute_confusion_matrix(self.y_true, self.y_pred, num_classes=5)
        self.assertEqual(cm.shape, (5, 5))
        self.assertEqual(cm.sum(), len(self.y_true))
        self.assertEqual(cm[1, 1], 3)
        self.assertEqual(cm[1, 2], 1)

    def test_iou_and_miou_perfect(self):
        cm = compute_confusion_matrix(self.y_true_perfect, self.y_pred_perfect, num_classes=5)
        ious = compute_iou_per_class(cm)
        for c in range(5):
            self.assertAlmostEqual(ious[c], 1.0)

        # Structural mIoU excluding background (classes 1..4)
        miou_struct = compute_miou(ious, include_background=False)
        self.assertAlmostEqual(miou_struct, 1.0)

        # All classes mIoU (classes 0..4)
        miou_all = compute_miou(ious, include_background=True)
        self.assertAlmostEqual(miou_all, 1.0)

        oa = compute_overall_accuracy(cm)
        self.assertAlmostEqual(oa, 1.0)

    def test_imperfect_metrics(self):
        cm = compute_confusion_matrix(self.y_true, self.y_pred, num_classes=5)
        ious = compute_iou_per_class(cm)
        self.assertAlmostEqual(ious[1], 0.75)
        self.assertAlmostEqual(ious[2], 2.0 / 3.0)

        accs = compute_class_accuracy(cm)
        self.assertAlmostEqual(accs[1], 3.0 / 4.0)
        self.assertAlmostEqual(accs[2], 1.0)

        f1s = compute_f1_scores(cm)
        self.assertIn("precision", f1s[1])
        self.assertIn("recall", f1s[1])
        self.assertIn("f1", f1s[1])


if __name__ == "__main__":
    unittest.main()
