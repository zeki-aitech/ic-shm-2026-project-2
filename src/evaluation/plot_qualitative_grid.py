"""
Assembles Figure 4 (Section 5.3): a grid of held-out views, each shown as four panels -
rendered RGB, real photograph, rendered semantic map, and ground-truth mask - colored by the
official per-class legend (`src.colmap_io.semantic_voting.CLASS_COLORS`).

Consumes the per-view PNGs already produced by `src.gaussian_splatting.render` (`{id}_rgb.png`,
`{id}_sem.png`, raw class-id grayscale) plus the dataset's own undistorted photo and GT mask for
each id - no GPU/model loading needed here, so this stays fast and easily testable.
"""
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.colmap_io.semantic_voting import CLASS_COLORS

NUM_CLASSES = 5
_LUT = np.stack([CLASS_COLORS[c] for c in range(NUM_CLASSES)]).astype(np.float32) / 255.0


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Maps a (H,W) array of class ids 0-4 to an (H,W,3) uint8 RGB image via the official
    per-class color legend."""
    return (_LUT[mask] * 255.0).astype(np.uint8)


def build_qualitative_grid(
    view_ids: List[str],
    render_dir: str,
    gt_images_dir: str,
    gt_masks_dir: str,
    output_path: str,
) -> str:
    """
    For each id in `view_ids`, loads `{render_dir}/{id}_rgb.png`, `{gt_images_dir}/{id}.png`,
    `{render_dir}/{id}_sem.png`, and `{gt_masks_dir}/{id}.png`, and arranges them into a
    len(view_ids) x 4 grid: Rendered RGB | Real Photo | Rendered Semantic | GT Mask.
    """
    col_titles = ["Rendered RGB", "Real Photo", "Rendered Semantic", "GT Mask"]
    n_rows = len(view_ids)

    fig, axes = plt.subplots(n_rows, 4, figsize=(4 * 3.0, n_rows * 2.25), dpi=200)
    if n_rows == 1:
        axes = axes[None, :]

    for row, vid in enumerate(view_ids):
        rendered_rgb = np.asarray(Image.open(os.path.join(render_dir, f"{vid}_rgb.png")).convert("RGB"))
        real_photo = np.asarray(Image.open(os.path.join(gt_images_dir, f"{vid}.png")).convert("RGB"))
        rendered_sem = colorize_mask(np.asarray(Image.open(os.path.join(render_dir, f"{vid}_sem.png"))))
        gt_mask = colorize_mask(np.asarray(Image.open(os.path.join(gt_masks_dir, f"{vid}.png"))))

        panels = [rendered_rgb, real_photo, rendered_sem, gt_mask]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=16, fontweight="bold")
            if col == 0:
                ax.set_ylabel(vid, fontsize=15, fontweight="bold", rotation=0, labelpad=32, va="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description="Assemble Figure 4 (qualitative render grid)")
    parser.add_argument("--view-ids", nargs="+", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument(
        "--gt-images-dir",
        default=os.path.join(project_root, "outputs", "undistorted_images"),
    )
    parser.add_argument(
        "--gt-masks-dir", default=os.path.join(project_root, "outputs", "gt_masks")
    )
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "paper", "figures", "fig4_qualitative_grid.png"),
    )
    args = parser.parse_args()

    out = build_qualitative_grid(
        args.view_ids, args.render_dir, args.gt_images_dir, args.gt_masks_dir, args.output
    )
    print(f"[plot_qualitative_grid] wrote {out}")


if __name__ == "__main__":
    main()
