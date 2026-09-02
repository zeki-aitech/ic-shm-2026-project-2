"""Evaluation module for IC-SHM 2026 Project 2."""
from .metrics import (
    compute_confusion_matrix,
    compute_iou_per_class,
    compute_miou,
    compute_overall_accuracy,
    compute_class_accuracy,
    compute_f1_scores,
    compute_deck_planarity_mad,
    compute_cable_fan_deviation,
    EvaluationReport,
)

__all__ = [
    "compute_confusion_matrix",
    "compute_iou_per_class",
    "compute_miou",
    "compute_overall_accuracy",
    "compute_class_accuracy",
    "compute_f1_scores",
    "compute_deck_planarity_mad",
    "compute_cable_fan_deviation",
    "EvaluationReport",
]
