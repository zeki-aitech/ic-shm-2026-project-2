"""
Evaluation metrics module for IC-SHM 2026 Project 2.

Implements the official evaluation framework defined in docs/EVALUATION_METRICS.md:
1. 2D / 3D Semantic Segmentation Metrics:
   - Confusion Matrix, per-class IoU, mIoU, Overall Accuracy (OA), Mean Accuracy (mAcc), F1-scores.
2. Structure-Aware SHM Geometric Metrics:
   - Deck Planarity Residuals (MAD - Median Absolute Deviation in meters).
   - Stay-Cable Fan Plane Alignment & Outlier Ratio.
3. Summary Report Generators:
   - Formats evaluation results into publication-ready Markdown tables and JSON summaries.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "deck",
    2: "stay_cable",
    3: "tower",
    4: "foundation",
}


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5
) -> np.ndarray:
    """
    Computes the (num_classes x num_classes) confusion matrix.

    Args:
        y_true: Ground-truth class labels (1D array of integers).
        y_pred: Predicted class labels (1D array of integers).
        num_classes: Total number of classes (default: 5).

    Returns:
        Confusion matrix M where M[i, j] is the count of samples with true class i
        and predicted class j.
    """
    y_true = np.asarray(y_true, dtype=np.int64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.int64).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        )

    # Filter out invalid class labels
    valid_mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

    indices = num_classes * y_t + y_p
    matrix = np.bincount(indices, minlength=num_classes**2).reshape((num_classes, num_classes))
    return matrix


def compute_iou_per_class(confusion_matrix: np.ndarray) -> Dict[int, float]:
    """
    Computes Intersection-over-Union (IoU) for each class from a confusion matrix.

    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    """
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    num_classes = cm.shape[0]
    ious: Dict[int, float] = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        ious[c] = float(tp / denom) if denom > 0 else 0.0

    return ious


def compute_miou(
    iou_dict: Dict[int, float], include_background: bool = False
) -> float:
    """
    Computes Mean IoU (mIoU) across structural classes.

    Args:
        iou_dict: Dictionary mapping class_id -> IoU value.
        include_background: If False, averages over classes 1..4 (bridge components).

    Returns:
        Mean IoU as a float.
    """
    classes_to_avg = (
        list(iou_dict.keys())
        if include_background
        else [c for c in iou_dict.keys() if c != 0]
    )

    if not classes_to_avg:
        return 0.0

    return float(np.mean([iou_dict[c] for c in classes_to_avg]))


def compute_overall_accuracy(confusion_matrix: np.ndarray) -> float:
    """
    Computes Overall Accuracy (OA) = sum(TP) / total_samples.
    """
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.trace(cm) / total)


def compute_class_accuracy(confusion_matrix: np.ndarray) -> Dict[int, float]:
    """
    Computes per-class accuracy (Recall / Sensitivity) = TP_c / (TP_c + FN_c).
    """
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    num_classes = cm.shape[0]
    accs: Dict[int, float] = {}

    for c in range(num_classes):
        tp = cm[c, c]
        total_true = cm[c, :].sum()
        accs[c] = float(tp / total_true) if total_true > 0 else 0.0

    return accs


def compute_f1_scores(confusion_matrix: np.ndarray) -> Dict[int, Dict[str, float]]:
    """
    Computes Precision, Recall, and F1-score for each class.
    """
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    num_classes = cm.shape[0]
    scores: Dict[int, Dict[str, float]] = {}

    for c in range(num_classes):
        tp = cm[c, c]
        pred_total = cm[:, c].sum()
        true_total = cm[c, :].sum()

        precision = float(tp / pred_total) if pred_total > 0 else 0.0
        recall = float(tp / true_total) if true_total > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        scores[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return scores


def compute_deck_planarity_mad(deck_points: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    Computes deck surface planarity using 2-pass PCA plane fitting and MAD.

    Args:
        deck_points: (N, 3) spatial coordinates of points labeled as 'deck'.

    Returns:
        (mad_residual, plane_normal, plane_d) where:
            mad_residual: Median Absolute Deviation of residuals (in meters).
            plane_normal: (3,) unit normal vector to the deck plane.
            plane_d: scalar plane offset (n . x + d = 0).
    """
    points = np.asarray(deck_points, dtype=np.float64)
    if len(points) < 3:
        return 0.0, np.array([0.0, 1.0, 0.0]), 0.0

    # Pass 1: Initial PCA Plane
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / len(points)
    _, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # Smallest eigenvalue corresponds to normal
    normal = normal / np.linalg.norm(normal)
    d = -float(normal @ centroid)

    # Compute signed residuals: r_i = n . x_i + d
    residuals = np.abs(points @ normal + d)
    med_res = np.median(residuals)
    mad_1 = np.median(np.abs(residuals - med_res))

    # Pass 2: Refined PCA on inliers (|r_i| <= 3 * MAD)
    inliers = points[residuals <= max(3.0 * mad_1, 0.02)]
    if len(inliers) >= 3:
        centroid2 = inliers.mean(axis=0)
        centered2 = inliers - centroid2
        cov2 = centered2.T @ centered2 / len(inliers)
        _, eigvecs2 = np.linalg.eigh(cov2)
        normal = eigvecs2[:, 0]
        normal = normal / np.linalg.norm(normal)
        d = -float(normal @ centroid2)
        residuals = np.abs(inliers @ normal + d)

    final_mad = float(np.median(residuals))
    return final_mad, normal, d


def compute_cable_fan_deviation(
    cable_points: np.ndarray,
    lateral_axis: np.ndarray,
    d_left: float,
    d_right: float,
    x0: Optional[np.ndarray] = None,
    tau_threshold: float = 0.10,
) -> Tuple[float, float]:
    """
    Measures how closely cable points align to the two tower-anchored fan planes.

    Args:
        cable_points: (N, 3) coordinates of points labeled as 'stay_cable'.
        lateral_axis: (3,) unit vector perpendicular to the bridge deck span (axis w).
        d_left: lateral offset of left cable fan sheet.
        d_right: lateral offset of right cable fan sheet.
        x0: reference origin coordinate (default: origin [0,0,0]).
        tau_threshold: maximum allowed deviation in meters (default: 0.10m).

    Returns:
        (mean_deviation, outlier_ratio) where:
            mean_deviation: Mean distance of cable points to nearest fan plane (in meters).
            outlier_ratio: Fraction of cable points exceeding tau_threshold.
    """
    points = np.asarray(cable_points, dtype=np.float64)
    if len(points) == 0:
        return 0.0, 0.0

    w = lateral_axis / np.linalg.norm(lateral_axis)
    origin = np.zeros(3) if x0 is None else np.asarray(x0)

    # Compute lateral projections: p_lat = (x - x0) . w
    p_lat = (points - origin) @ w

    # Distance to nearest fan plane: min(|p_lat - d_left|, |p_lat - d_right|)
    dist_left = np.abs(p_lat - d_left)
    dist_right = np.abs(p_lat - d_right)
    min_dist = np.minimum(dist_left, dist_right)

    mean_dev = float(np.mean(min_dist))
    outlier_ratio = float(np.mean(min_dist > tau_threshold))
    return mean_dev, outlier_ratio


@dataclass
class EvaluationReport:
    """Container for comprehensive evaluation results."""

    confusion_matrix: np.ndarray
    ious: Dict[int, float]
    miou_structural: float
    miou_all: float
    overall_accuracy: float
    class_accuracies: Dict[int, float]
    f1_scores: Dict[int, Dict[str, float]]
    deck_planarity_mad: Optional[float] = None
    cable_mean_deviation: Optional[float] = None
    cable_outlier_ratio: Optional[float] = None
    mean_reprojection_error: Optional[float] = None

    def to_markdown(self) -> str:
        """Renders the evaluation report as a publication-ready Markdown table."""
        lines = [
            "### 📊 IC-SHM 2026 Project 2 — Evaluation Performance Report",
            "",
            "#### 1. Semantic Segmentation Benchmarks (3D / 2D)",
            "| Class ID | Component Name | IoU (%) | Precision (%) | Recall (%) | F1-Score (%) |",
            "| :---: | :--- | :---: | :---: | :---: | :---: |",
        ]

        for cid, cname in CLASS_NAMES.items():
            iou_val = self.ious.get(cid, 0.0) * 100.0
            prec_val = self.f1_scores.get(cid, {}).get("precision", 0.0) * 100.0
            rec_val = self.f1_scores.get(cid, {}).get("recall", 0.0) * 100.0
            f1_val = self.f1_scores.get(cid, {}).get("f1", 0.0) * 100.0
            lines.append(
                f"| **{cid}** | `{cname}` | {iou_val:6.2f}% | {prec_val:6.2f}% | {rec_val:6.2f}% | {f1_val:6.2f}% |"
            )

        lines.extend([
            "",
            f"- **Structural mIoU (Classes 1–4)**: **`{self.miou_structural * 100:.2f}%`**",
            f"- **Mean IoU (All 5 Classes)**: `{self.miou_all * 100:.2f}%`",
            f"- **Overall Accuracy (OA)**: `{self.overall_accuracy * 100:.2f}%`",
            f"- **Cable IoU (Key Target)**: **`{self.ious.get(2, 0.0) * 100:.2f}%`**",
            "",
            "#### 2. Domain-Specific Structural Health Monitoring (SHM) Metrics",
            "| SHM Metric | Value | Reference Standard | Assessment |",
            "| :--- | :---: | :---: | :---: |",
        ])

        if self.deck_planarity_mad is not None:
            status = "✅ PASS" if self.deck_planarity_mad < 0.05 else "⚠️ NEEDS TUNING"
            lines.append(
                f"| **Deck Planarity Residual (MAD)** | `{self.deck_planarity_mad:.4f} m` | `< 0.05 m` | {status} |"
            )

        if self.cable_mean_deviation is not None:
            status = "✅ PASS" if self.cable_mean_deviation < 0.10 else "⚠️ NEEDS TUNING"
            lines.append(
                f"| **Cable Fan Sheet Deviation** | `{self.cable_mean_deviation:.4f} m` | `< 0.10 m` | {status} |"
            )

        if self.cable_outlier_ratio is not None:
            status = "✅ PASS" if self.cable_outlier_ratio < 0.02 else "⚠️ NEEDS TUNING"
            lines.append(
                f"| **Off-Fan Cable Outlier Ratio** | `{self.cable_outlier_ratio * 100:.2f}%` | `< 2.00%` | {status} |"
            )

        if self.mean_reprojection_error is not None:
            status = "✅ PASS" if self.mean_reprojection_error < 1.0 else "⚠️ NEEDS TUNING"
            lines.append(
                f"| **Mean Reprojection Error** | `{self.mean_reprojection_error:.2f} px` | `< 1.00 px` | {status} |"
            )

        return "\n".join(lines)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    deck_points: Optional[np.ndarray] = None,
    cable_points: Optional[np.ndarray] = None,
    lateral_axis: Optional[np.ndarray] = None,
    d_left: Optional[float] = None,
    d_right: Optional[float] = None,
    mean_reproj_error: Optional[float] = None,
) -> EvaluationReport:
    """
    Executes a complete evaluation pipeline on predicted vs. true labels and spatial coordinates.
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=5)
    ious = compute_iou_per_class(cm)
    miou_struct = compute_miou(ious, include_background=False)
    miou_all = compute_miou(ious, include_background=True)
    oa = compute_overall_accuracy(cm)
    accs = compute_class_accuracy(cm)
    f1s = compute_f1_scores(cm)

    deck_mad = None
    if deck_points is not None and len(deck_points) >= 3:
        deck_mad, _, _ = compute_deck_planarity_mad(deck_points)

    cable_dev = None
    cable_outliers = None
    if (
        cable_points is not None
        and len(cable_points) > 0
        and lateral_axis is not None
        and d_left is not None
        and d_right is not None
    ):
        cable_dev, cable_outliers = compute_cable_fan_deviation(
            cable_points, lateral_axis, d_left, d_right
        )

    return EvaluationReport(
        confusion_matrix=cm,
        ious=ious,
        miou_structural=miou_struct,
        miou_all=miou_all,
        overall_accuracy=oa,
        class_accuracies=accs,
        f1_scores=f1s,
        deck_planarity_mad=deck_mad,
        cable_mean_deviation=cable_dev,
        cable_outlier_ratio=cable_outliers,
        mean_reprojection_error=mean_reproj_error,
    )
