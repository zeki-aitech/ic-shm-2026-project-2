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
    compute_deck_planarity_mad,
    compute_cable_fan_deviation,
    evaluate_predictions,
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
        # True deck=1 (4 samples): pred deck=3, pred cable=1
        self.assertEqual(cm[1, 1], 3)
        self.assertEqual(cm[1, 2], 1)

    def test_iou_and_miou_perfect(self):
        cm = compute_confusion_matrix(self.y_true_perfect, self.y_pred_perfect, num_classes=5)
        ious = compute_iou_per_class(cm)
        for c in range(5):
            self.assertAlmostEqual(ious[c], 1.0)

        miou_struct = compute_miou(ious, include_background=False)
        self.assertAlmostEqual(miou_struct, 1.0)

        oa = compute_overall_accuracy(cm)
        self.assertAlmostEqual(oa, 1.0)

    def test_imperfect_metrics(self):
        cm = compute_confusion_matrix(self.y_true, self.y_pred, num_classes=5)
        ious = compute_iou_per_class(cm)
        # Deck: TP=3, FP=0, FN=1 -> IoU = 3/4 = 0.75
        self.assertAlmostEqual(ious[1], 0.75)
        # Cable: TP=2, FP=1, FN=0 -> IoU = 2/3 = 0.6667
        self.assertAlmostEqual(ious[2], 2.0 / 3.0)

        accs = compute_class_accuracy(cm)
        self.assertAlmostEqual(accs[1], 3.0 / 4.0)
        self.assertAlmostEqual(accs[2], 1.0)

        f1s = compute_f1_scores(cm)
        self.assertIn("precision", f1s[1])
        self.assertIn("recall", f1s[1])
        self.assertIn("f1", f1s[1])

    def test_deck_planarity_mad(self):
        np.random.seed(42)
        # Create a planar horizontal deck at Y = 0 with 2D spatial extent
        n_pts = 200
        x = np.random.uniform(-20, 20, n_pts)
        z = np.random.uniform(-3, 3, n_pts)
        y = np.random.normal(0, 0.005, n_pts) # 5mm noise
        deck_pts = np.column_stack([x, y, z])

        mad, normal, d = compute_deck_planarity_mad(deck_pts)
        self.assertLess(mad, 0.05) # Must satisfy SHM target < 0.05m
        # Normal should align closely with Y-axis [0, 1, 0]
        self.assertGreater(abs(normal[1]), 0.95)

    def test_cable_fan_deviation(self):
        np.random.seed(42)
        # Create cable points along lateral axis at d_left = -2.0, d_right = +2.0
        n_pts = 50
        lateral_axis = np.array([0.0, 0.0, 1.0])
        x = np.random.uniform(-10, 10, n_pts)
        y = np.random.uniform(5, 20, n_pts)
        z = np.array([-2.0] * 25 + [2.0] * 25) + np.random.normal(0, 0.02, n_pts)
        cable_pts = np.column_stack([x, y, z])

        mean_dev, outlier_ratio = compute_cable_fan_deviation(
            cable_pts, lateral_axis, d_left=-2.0, d_right=2.0, tau_threshold=0.10
        )
        self.assertLess(mean_dev, 0.05)
        self.assertEqual(outlier_ratio, 0.0)

    def test_full_evaluation_report_markdown(self):
        report = evaluate_predictions(
            self.y_true, self.y_pred, mean_reproj_error=0.65
        )
        md = report.to_markdown()
        self.assertIn("Structural mIoU", md)
        self.assertIn("Mean Reprojection Error", md)
        self.assertIn("PASS", md)


if __name__ == "__main__":
    unittest.main()
