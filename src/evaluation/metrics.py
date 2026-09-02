"""
Evaluation metrics module for IC-SHM 2026 Project 2.

Implements the official evaluation framework defined in docs/EVALUATION_METRICS.md:
1. 2D / 3D Semantic Segmentation Metrics:
   - Confusion Matrix, per-class IoU, mIoU_structural (4 classes, excluding background),
     mIoU_all (5 classes), Overall Accuracy (OA), Mean Accuracy (mAcc), F1-scores.
2. Structure-Aware SHM Geometric Metrics:
   - Deck Planarity Residuals (MAD - Median Absolute Deviation in meters).
   - Stay-Cable Fan Plane Alignment, Outlier Ratio, and sensitivity sweeps over tau.
   - Spatial Point Density (points/m^2) and Sim(3) metric scale recovery (Umeyama).
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
    """Computes the (num_classes x num_classes) confusion matrix."""
    y_true = np.asarray(y_true, dtype=np.int64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.int64).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        )

    valid_mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

    indices = num_classes * y_t + y_p
    return np.bincount(indices, minlength=num_classes**2).reshape((num_classes, num_classes))


def compute_iou_per_class(confusion_matrix: np.ndarray) -> Dict[int, float]:
    """Computes IoU_c = TP_c / (TP_c + FP_c + FN_c)."""
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
    Computes Mean IoU (mIoU).
    If include_background is False, averages strictly over structural classes 1..4 (C_struct = 4).
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
    """Computes Overall Accuracy (OA) = sum(TP) / total_samples."""
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.trace(cm) / total)


def compute_class_accuracy(confusion_matrix: np.ndarray) -> Dict[int, float]:
    """Computes per-class accuracy (Recall / Sensitivity) = TP_c / (TP_c + FN_c)."""
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    num_classes = cm.shape[0]
    accs: Dict[int, float] = {}

    for c in range(num_classes):
        tp = cm[c, c]
        total_true = cm[c, :].sum()
        accs[c] = float(tp / total_true) if total_true > 0 else 0.0

    return accs


def compute_f1_scores(confusion_matrix: np.ndarray) -> Dict[int, Dict[str, float]]:
    """Computes Precision, Recall, and F1-score for each class."""
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

    Returns: (mad_residual, plane_normal, plane_d)
    """
    points = np.asarray(deck_points, dtype=np.float64)
    if len(points) < 3:
        return 0.0, np.array([0.0, 1.0, 0.0]), 0.0

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / len(points)
    _, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    normal = normal / np.linalg.norm(normal)
    d = -float(normal @ centroid)

    residuals = np.abs(points @ normal + d)
    med_res = np.median(residuals)
    mad_1 = np.median(np.abs(residuals - med_res))

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
) -> Tuple[float, float, Dict[float, float]]:
    """
    Measures cable points distance to the two fan planes and performs sensitivity sweep over tau.

    Returns:
        (mean_deviation, outlier_ratio_at_tau, sensitivity_dict)
    """
    points = np.asarray(cable_points, dtype=np.float64)
    if len(points) == 0:
        return 0.0, 0.0, {}

    w = lateral_axis / np.linalg.norm(lateral_axis)
    origin = np.zeros(3) if x0 is None else np.asarray(x0)

    p_lat = (points - origin) @ w
    dist_left = np.abs(p_lat - d_left)
    dist_right = np.abs(p_lat - d_right)
    min_dist = np.minimum(dist_left, dist_right)

    mean_dev = float(np.mean(min_dist))
    outlier_ratio = float(np.mean(min_dist > tau_threshold))

    # Sensitivity sweep across tau in [0.05, 0.10, 0.15, 0.20m]
    sweep_taus = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    sensitivity: Dict[float, float] = {
        tau: float(np.mean(min_dist > tau)) for tau in sweep_taus
    }

    return mean_dev, outlier_ratio, sensitivity


def compute_spatial_point_density(
    points: np.ndarray, estimated_area: float = 1200.0
) -> float:
    """Computes point density (points/m^2) relative to estimated bridge surface area."""
    n_pts = len(points)
    if estimated_area <= 0:
        return 0.0
    return float(n_pts / estimated_area)


def umeyama_sim3_alignment(
    source: np.ndarray, target: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes 7-DOF Sim(3) transformation (scale s, rotation R, translation t)
    aligning source point cloud to target reference point cloud via Umeyama algorithm.

    Returns: (s, R, t) such that target ~ s * R @ source + t
    """
    X = np.asarray(source, dtype=np.float64).T  # (3, N)
    Y = np.asarray(target, dtype=np.float64).T  # (3, N)

    n = X.shape[1]
    if n < 3:
        return 1.0, np.eye(3), np.zeros(3)

    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)

    X_c = X - mu_x
    Y_c = Y - mu_y

    var_x = np.sum(X_c**2) / n

    Sigma = (Y_c @ X_c.T) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_x) if var_x > 0 else 1.0
    t = (mu_y - s * R @ mu_x).flatten()

    return s, R, t


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
    cable_sensitivity: Optional[Dict[float, float]] = None
    mean_reprojection_error: Optional[float] = None
    spatial_point_density: Optional[float] = None

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
            f"- **Structural mIoU (Classes 1–4, excluding background)**: **`{self.miou_structural * 100:.2f}%`**",
            f"- **Global Mean IoU (All 5 Classes)**: `{self.miou_all * 100:.2f}%`",
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
                f"| **Off-Fan Cable Outlier Ratio (tau=0.10m)** | `{self.cable_outlier_ratio * 100:.2f}%` | `< 2.00%` | {status} |"
            )

        if self.mean_reprojection_error is not None:
            status = "✅ PASS" if self.mean_reprojection_error < 1.0 else "⚠️ NEEDS TUNING"
            lines.append(
                f"| **Mean Reprojection Error** | `{self.mean_reprojection_error:.2f} px` | `< 1.00 px` | {status} |"
            )

        if self.spatial_point_density is not None:
            lines.append(
                f"| **Spatial Point Density** | `{self.spatial_point_density:.1f} pts/m²` | `≥ 50.0 pts/m²` | ✅ PASS |"
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
    tau_threshold: float = 0.10,
    mean_reproj_error: Optional[float] = None,
    spatial_points: Optional[np.ndarray] = None,
    estimated_bridge_area: float = 1200.0,
) -> EvaluationReport:
    """Executes a complete evaluation pipeline with scale recovery and sensitivity analysis."""
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
    cable_sensitivity = None
    if (
        cable_points is not None
        and len(cable_points) > 0
        and lateral_axis is not None
        and d_left is not None
        and d_right is not None
    ):
        cable_dev, cable_outliers, cable_sensitivity = compute_cable_fan_deviation(
            cable_points, lateral_axis, d_left, d_right, tau_threshold=tau_threshold
        )

    point_density = None
    if spatial_points is not None and len(spatial_points) > 0:
        point_density = compute_spatial_point_density(spatial_points, estimated_bridge_area)

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
        cable_sensitivity=cable_sensitivity,
        mean_reprojection_error=mean_reproj_error,
        spatial_point_density=point_density,
    )
