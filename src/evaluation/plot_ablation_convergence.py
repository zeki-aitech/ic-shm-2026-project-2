"""
Figure 9 (Section 5.2, optional): tracks stay_cable IoU on the 60-view holdout across training
steps for three real checkpoints - the baseline (strict-majority cable voting + semantic
warm-start), the plain-plurality ablation, and the no-semantic-warmstart ablation - to check
whether the Table 3 result (no measurable *final*-IoU difference between the three) holds
throughout training or only appears once training has converged.

Deliberately skips PSNR/SSIM/LPIPS (unlike `render_metrics.py`): this figure only tracks
semantic IoU across many checkpoints per run, and skipping the LPIPS forward pass makes that
several times faster with no loss of relevance, since visual fidelity is not what's being
studied here.
"""
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import CLASS_NAMES, compute_confusion_matrix, compute_iou_per_class


def evaluate_semantic_iou(checkpoint_path: str, holdout_cameras: List, device: str = "cuda") -> Dict[int, float]:
    """Loads a checkpoint and returns per-class IoU on `holdout_cameras`, skipping the visual
    (PSNR/SSIM/LPIPS) metrics computed by `render_metrics.evaluate_render_holdout`."""
    import torch
    from PIL import Image

    from src.gaussian_splatting.model import SemanticGaussianModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SemanticGaussianModel.from_state_dict(ckpt["params"], device=device)

    conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for camera in holdout_cameras:
        with torch.no_grad():
            _rgb, sem_logits = model.render(camera)
        pred_mask = sem_logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
        gt_mask = np.asarray(Image.open(camera.mask_path))
        conf += compute_confusion_matrix(gt_mask.ravel(), pred_mask.ravel(), len(CLASS_NAMES))

    return compute_iou_per_class(conf)


def collect_convergence_curves(
    checkpoint_dirs: Dict[str, str], steps: List[int], holdout_cameras: List, device: str = "cuda"
) -> Dict[str, Dict[int, Dict[int, float]]]:
    """For each (label -> checkpoint_dir) and each step in `steps`, evaluates
    `{checkpoint_dir}/step_{step}.pt` and returns {label: {step: {class_id: iou}}}."""
    curves: Dict[str, Dict[int, Dict[int, float]]] = {}
    for label, ckpt_dir in checkpoint_dirs.items():
        curves[label] = {}
        for step in steps:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pt")
            curves[label][step] = evaluate_semantic_iou(ckpt_path, holdout_cameras, device=device)
    return curves


def plot_ablation_convergence(
    curves: Dict[str, Dict[int, Dict[int, float]]], class_id: int, output_path: str
) -> str:
    """Plots `class_id`'s IoU vs. training step, one line per label in `curves`."""
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    colors = {"baseline": "#1a1a1a", "plain_plurality": "#c0392b", "no_warmstart": "#2166ac"}
    labels_display = {
        "baseline": "Baseline (strict-majority + warm-start)",
        "plain_plurality": "Plain plurality (no strict-majority rule)",
        "no_warmstart": "No semantic warm-start (neutral init)",
    }

    for label, per_step in curves.items():
        steps = sorted(per_step.keys())
        ious = [per_step[s][class_id] * 100 for s in steps]
        ax.plot(
            steps, ious, marker="o", markersize=4, linewidth=2.2,
            color=colors.get(label), label=labels_display.get(label, label),
        )

    ax.set_xlabel("Training step", fontsize=13)
    ax.set_ylabel(f"{CLASS_NAMES.get(class_id, class_id)} IoU (%)", fontsize=13)
    ax.set_title("Cable IoU convergence: ablation vs. baseline", fontsize=14.5, fontweight="bold")
    ax.legend(fontsize=10.5, loc="lower right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description="Track cable IoU vs. training step across ablation runs")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--plain-plurality-dir", required=True)
    parser.add_argument("--no-warmstart-dir", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10000, 16000, 24000, 32000, 40000])
    parser.add_argument("--colmap-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--unlabeled-dir", default=None)
    parser.add_argument("--gt-masks-dir", default=None)
    parser.add_argument("--undistorted-dir", default=None)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument(
        "--output", default=os.path.join(project_root, "paper", "figures", "fig9_ablation_convergence.png")
    )
    args = parser.parse_args()

    import torch

    from src.gaussian_splatting.train import prepare_training_data

    dataset_dir = os.getenv("CONTEST_DATASET_DIR", os.path.join(project_root, "data", "Contest Dataset"))
    colmap_dir = args.colmap_dir or os.path.join(dataset_dir, "camera_parameters")
    images_dir = args.images_dir or os.path.join(dataset_dir, "images")
    unlabeled_dir = args.unlabeled_dir or os.path.join(dataset_dir, "unlabeled_Images")
    gt_masks_dir = args.gt_masks_dir or os.path.join(project_root, "outputs", "gt_masks")
    undistorted_dir = args.undistorted_dir or os.path.join(project_root, "outputs", "undistorted_images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, _, _, _, holdout_cameras, _ = prepare_training_data(
        colmap_dir, images_dir, unlabeled_dir, gt_masks_dir, None, undistorted_dir, args.holdout_ratio
    )

    checkpoint_dirs = {
        "baseline": args.baseline_dir,
        "plain_plurality": args.plain_plurality_dir,
        "no_warmstart": args.no_warmstart_dir,
    }
    curves = collect_convergence_curves(checkpoint_dirs, args.steps, holdout_cameras, device=device)

    STAY_CABLE_CLASS_ID = 2
    out = plot_ablation_convergence(curves, STAY_CABLE_CLASS_ID, args.output)
    print(f"[plot_ablation_convergence] wrote {out}")
    for label, per_step in curves.items():
        for step in sorted(per_step.keys()):
            print(f"  {label:16s} step={step:6d} cable_iou={per_step[step][STAY_CABLE_CLASS_ID]*100:.2f}%")


if __name__ == "__main__":
    main()
