"""
Renders Figure 2 (Section 3.4): a two-panel illustration of the background-bleeding problem in
cable annotations and the strict-majority voting rule that counteracts it.

Panel A is real data: a crop of an actual undistorted UAV image with its ground-truth mask
overlaid (official per-class colors), showing that the `stay_cable` polygon covers large regions
of sky/water rather than just the thin cable strands - not a staged example.

Panel B is a schematic (not tied to one specific real 3D point) illustrating the multi-view
voting mechanism: several cameras observe a 3D point, a minority of them see it as cable, and
because that falls short of an absolute majority, the strict-majority rule (Section 3.4) falls
back to plurality among the non-cable votes.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image

from src.colmap_io.semantic_voting import CLASS_COLORS, CLASS_NAMES

_LUT = np.stack([CLASS_COLORS[c] for c in range(len(CLASS_NAMES))]).astype(np.float32)


def render_overlay_crop(image_path: str, mask_path: str, crop_box, alpha: float = 0.55) -> np.ndarray:
    """`crop_box`: (y0, y1, x0, x1). Returns an (H,W,3) uint8 image/mask overlay, background
    class left untinted so the real photo shows through."""
    img = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float32)
    mask = np.asarray(Image.open(mask_path))
    y0, y1, x0, x1 = crop_box
    img_c, mask_c = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    overlay = img_c.copy()
    for c in range(1, len(CLASS_NAMES)):  # skip background (0): leave the real photo untinted
        m = mask_c == c
        overlay[m] = (1 - alpha) * overlay[m] + alpha * _LUT[c]
    return overlay.astype(np.uint8)


def _draw_camera(ax, xy, angle_deg, label, vote_color, vote_text):
    """Draws a small camera glyph (triangle) at `xy` pointing toward the origin, with its
    observed-class vote annotated."""
    x, y = xy
    ax.plot(x, y, marker=(3, 0, angle_deg), markersize=16, color="dimgray", zorder=4)
    ax.text(x, y - 0.55, label, ha="center", va="top", fontsize=8, color="dimgray")
    ax.text(
        x, y - 0.95, vote_text, ha="center", va="top", fontsize=8.5, fontweight="bold",
        color=vote_color,
    )


def draw_voting_schematic(ax):
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.6, 4.2)
    ax.axis("off")
    ax.set_title("Illustrative example: multi-view vote for one 3D point", fontsize=10, pad=8)

    origin = (0.0, 0.5)
    ax.plot(*origin, marker="o", markersize=9, color="black", zorder=5)
    ax.text(0, 1.0, "3D point", ha="center", fontsize=9, fontweight="bold")

    cable_rgb = tuple(_LUT[2] / 255.0)
    bg_rgb = tuple(_LUT[0] / 255.0)
    deck_rgb = tuple(_LUT[1] / 255.0)

    cams = [
        # (x, y, angle_deg_pointing_to_origin, label, observed_class, color, is_cable)
        (-3.4, 2.6, -50, "view 1", "cable", cable_rgb),
        (0.0, 3.8, 180, "view 2", "cable", cable_rgb),
        (3.4, 2.6, 130, "view 3", "background", bg_rgb),
        (-2.6, -3.0, 60, "view 4", "background", bg_rgb),
        (2.6, -3.0, 100, "view 5", "deck", deck_rgb),
    ]
    for x, y, angle, label, cls_text, color in cams:
        ax.add_patch(
            FancyArrowPatch((x, y), origin, arrowstyle="-", linestyle=(0, (3, 3)),
                             color="lightgray", linewidth=1.1, zorder=1)
        )
        _draw_camera(ax, (x, y), angle, label, color, cls_text)

    ax.text(
        0, -3.9,
        "2/5 cable votes (40%) < 50% threshold\n"
        "→ cable votes discarded → plurality among remaining: background (2) vs. deck (1)\n"
        "→ point labeled background",
        ha="center", va="top", fontsize=8.3, color="black",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="gray"),
    )


def build_cable_voting_figure(image_path: str, mask_path: str, crop_box, output_path: str) -> str:
    overlay = render_overlay_crop(image_path, mask_path, crop_box)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=200, gridspec_kw={"width_ratios": [1.1, 1]})

    axA.imshow(overlay)
    axA.set_xticks([])
    axA.set_yticks([])
    axA.set_title("Real example: GT mask overlaid on an undistorted UAV image", fontsize=10, pad=8)
    axA.set_xlabel(
        "cyan = stay_cable; note the polygon covers sky and river far beyond the cable strands",
        fontsize=8.5, color="dimgray",
    )

    draw_voting_schematic(axB)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description="Render Figure 2 (cable background-bleeding + voting)")
    parser.add_argument(
        "--image", default=os.path.join(project_root, "outputs", "undistorted_images", "300.png")
    )
    parser.add_argument(
        "--mask", default=os.path.join(project_root, "outputs", "gt_masks", "300.png")
    )
    parser.add_argument("--crop", nargs=4, type=int, default=[0, 780, 0, 950], metavar=("Y0", "Y1", "X0", "X1"))
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "paper", "figures", "fig2_cable_voting.png"),
    )
    args = parser.parse_args()

    out = build_cable_voting_figure(args.image, args.mask, tuple(args.crop), args.output)
    print(f"[plot_cable_voting_figure] wrote {out}")


if __name__ == "__main__":
    main()
