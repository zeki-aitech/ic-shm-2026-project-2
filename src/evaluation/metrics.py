"""
Semantic segmentation evaluation metrics for IC-SHM 2026 Project 2: confusion matrix, per-class
IoU, mIoU (structural, excluding background), overall accuracy, and per-class F1 scores over the
5 bridge classes. Also provides `trajectory_interleaved_split`, the shared train/holdout
partitioning used by both 2D segmentation training and Gaussian Splatting training/evaluation.
"""
from typing import Dict, List, Tuple
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


def trajectory_interleaved_split(
    sorted_ids: List,
    holdout_ratio: float = 0.20,
) -> Tuple[List, List]:
    """
    Partitions a trajectory-ordered list of ids into (train_ids, holdout_ids) by strided
    sampling: every Nth id (N = round(1/holdout_ratio)) is held out, the rest is train.

    `sorted_ids` must already be sorted along the UAV flight trajectory (e.g. by image name/ID)
    so the held-out ids are spread uniformly across the flight path rather than clustered.

    This is the single source of truth for the train/holdout partition shared by 2D segmentation
    training, Gaussian Splatting training, and render-based evaluation.
    """
    step = max(1, int(round(1.0 / holdout_ratio))) if holdout_ratio > 0 else len(sorted_ids) + 1
    holdout_ids = list(sorted_ids[step - 1 :: step])
    holdout_set = set(holdout_ids)
    train_ids = [i for i in sorted_ids if i not in holdout_set]
    return train_ids, holdout_ids
