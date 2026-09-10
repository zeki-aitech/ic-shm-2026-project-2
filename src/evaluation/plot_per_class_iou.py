"""
Renders the per-class IoU bar chart used as Figure 3 in the paper (Section 5.1), from the same
`iou_per_class` dict `render_metrics.RenderEvalReport` already produces - so the figure is always
regenerated from a real evaluation run rather than hand-typed numbers.
"""
import os
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import CLASS_NAMES, compute_miou

BACKGROUND_CLASS_ID = 0

# Official per-class colors (src/colmap_io/semantic_voting.py CLASS_COLORS), darkened slightly
# where needed for legibility as a solid bar on a white background.
PLOT_COLORS = {
    "deck": (1.0, 0.0, 0.0),
    "stay_cable": (0.0, 200 / 255, 200 / 255),
    "tower": (0.0, 180 / 255, 0.0),
    "foundation": (210 / 255, 180 / 255, 0.0),
}


def plot_per_class_iou(
    iou_per_class: Dict[int, float],
    output_path: str,
    structural_miou: Optional[float] = None,
    n_holdout_views: int = 30,
) -> str:
    """
    `iou_per_class`: class_id -> IoU in [0, 1] (as returned by `compute_iou_per_class`).
    `structural_miou`: if not given, computed from `iou_per_class` via `compute_miou`
    (background excluded), matching how the paper's reported mIoU is computed.
    """
    if structural_miou is None:
        structural_miou = compute_miou(iou_per_class, include_background=False)

    class_ids = sorted(cid for cid in iou_per_class if cid != BACKGROUND_CLASS_ID)
    names = [CLASS_NAMES[cid] for cid in class_ids]
    values = [iou_per_class[cid] * 100.0 for cid in class_ids]
    colors = [PLOT_COLORS.get(name, (0.5, 0.5, 0.5)) for name in names]

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.8, width=0.6, zorder=3)

    miou_pct = structural_miou * 100.0

    for bar, val in zip(bars, values):
        # Nudge a value label further above its bar when it would otherwise sit right on top
        # of the mIoU dashed line (close enough in height that the bold label text and the line
        # visually collide).
        offset = 1.3 if abs(val - miou_pct) < 1.5 else 0.6
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + offset, f"{val:.2f}%",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="black", zorder=4,
        )

    ax.axhline(miou_pct, color="black", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        len(names) - 0.38, miou_pct + 0.6, f"structural mIoU = {miou_pct:.2f}%",
        ha="right", va="bottom", fontsize=9, style="italic", color="dimgray", zorder=4,
    )

    lo = max(0.0, min(values) - 10.0)
    ax.set_ylim(lo, 100)
    ax.set_ylabel("IoU (%)", fontsize=11)
    ax.set_title(f"Per-Class IoU on the {n_holdout_views}-View Holdout", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    import argparse

    import torch

    from src.evaluation.render_metrics import evaluate_render_holdout
    from src.gaussian_splatting.model import SemanticGaussianModel
    from src.gaussian_splatting.train import prepare_training_data

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(
        description="Regenerate Figure 3 (per-class IoU bar chart) from a trained checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--colmap-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--unlabeled-dir", default=None)
    parser.add_argument("--gt-masks-dir", default=None)
    parser.add_argument("--undistorted-dir", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "paper", "figures", "fig3_per_class_iou.png"),
    )
    args = parser.parse_args()

    dataset_dir = os.getenv("CONTEST_DATASET_DIR", os.path.join(project_root, "data", "Contest Dataset"))
    colmap_dir = args.colmap_dir or os.path.join(dataset_dir, "camera_parameters")
    images_dir = args.images_dir or os.path.join(dataset_dir, "images")
    unlabeled_dir = args.unlabeled_dir or os.path.join(dataset_dir, "unlabeled_Images")
    gt_masks_dir = args.gt_masks_dir or os.path.join(project_root, "outputs", "gt_masks")
    undistorted_dir = args.undistorted_dir or os.path.join(project_root, "outputs", "undistorted_images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, _, _, _, holdout_cameras, _ = prepare_training_data(
        colmap_dir, images_dir, unlabeled_dir, gt_masks_dir, None, undistorted_dir,
        args.val_ratio, args.test_ratio,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SemanticGaussianModel.from_state_dict(ckpt["params"], device=device)

    report = evaluate_render_holdout(model, holdout_cameras, device=device)
    out_path = plot_per_class_iou(report.iou_per_class, args.output, n_holdout_views=len(holdout_cameras))
    print(f"[plot_per_class_iou] wrote {out_path}")


if __name__ == "__main__":
    main()
