"""
Renders Figure 3 (Section 3.4): a two-panel illustration of the coarse `stay_cable` annotation
region and the multi-view plurality voting mechanism used to initialize each Gaussian's semantic
logits from it.

Panel A is real data: a crop of an actual undistorted UAV image with its ground-truth mask
overlaid (official per-class colors), showing that the `stay_cable` polygon covers large regions
of sky/water rather than just the thin cable strands - not a staged example.

Panel B is a schematic (not tied to one specific real 3D point) illustrating the voting
mechanism: several cameras observe a 3D point and vote by simple plurality, with ties broken by
a fixed priority that favors thin/rare structural classes over background.
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


def _camera_marker_angle_deg(xy, target):
    """Matplotlib's regular-polygon marker angle is measured counterclockwise from the +y axis
    (angle=0 -> apex points up/+y; angle=90 -> apex points left/-x - verified empirically).
    Returns the angle that makes the triangle's apex point from `xy` toward `target`."""
    dx, dy = target[0] - xy[0], target[1] - xy[1]
    return np.degrees(np.arctan2(dy, dx)) - 90.0


def _draw_camera(ax, xy, target, label, vote_color, vote_text):
    """Draws a small camera glyph (triangle) at `xy`, its apex pointing toward `target` - i.e.
    the triangle's orientation shows which way that camera is looking - with its observed-class
    vote annotated below it."""
    x, y = xy
    angle_deg = _camera_marker_angle_deg(xy, target)
    ax.plot(x, y, marker=(3, 0, angle_deg), markersize=20, color="dimgray", zorder=4)
    ax.text(x, y - 0.6, label, ha="center", va="top", fontsize=11.5, color="dimgray")
    ax.text(
        x, y - 1.05, vote_text, ha="center", va="top", fontsize=12.5, fontweight="bold",
        color=vote_color,
    )


def draw_voting_schematic(ax):
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-6.3, 4.2)
    ax.axis("off")
    ax.set_title("Illustrative example: multi-view vote for one 3D point", fontsize=13.5, pad=10)

    origin = (0.0, 0.5)
    ax.plot(*origin, marker="o", markersize=11, color="black", zorder=5)
    ax.text(0, 1.05, "3D point", ha="center", fontsize=12.5, fontweight="bold")

    cable_rgb = tuple(_LUT[2] / 255.0)
    bg_rgb = tuple(_LUT[0] / 255.0)
    deck_rgb = tuple(_LUT[1] / 255.0)

    cams = [
        # (x, y, label, observed_class, color) - triangle apex is computed to point at `origin`
        (-3.4, 2.6, "view 1", "cable", cable_rgb),
        (0.0, 3.8, "view 2", "cable", cable_rgb),
        (3.4, 2.6, "view 3", "background", bg_rgb),
        (-2.6, -2.2, "view 4", "background", bg_rgb),
        (2.6, -2.2, "view 5", "deck", deck_rgb),
    ]
    for x, y, label, cls_text, color in cams:
        ax.add_patch(
            FancyArrowPatch((x, y), origin, arrowstyle="-", linestyle=(0, (3, 3)),
                             color="lightgray", linewidth=1.1, zorder=1)
        )
        _draw_camera(ax, (x, y), origin, label, color, cls_text)

    ax.text(
        0, -4.5,
        "cable (2) ties background (2), deck (1) trails\n"
        "→ plurality tie broken by fixed priority (thin/rare structures first)\n"
        "→ point labeled cable",
        ha="center", va="top", fontsize=11.5, color="black",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="gray"),
    )


def build_cable_voting_figure(image_path: str, mask_path: str, crop_box, output_path: str) -> str:
    overlay = render_overlay_crop(image_path, mask_path, crop_box)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.4), dpi=200, gridspec_kw={"width_ratios": [1.1, 1]})

    axA.imshow(overlay)
    axA.set_xticks([])
    axA.set_yticks([])
    axA.set_title("Real example: GT mask overlaid on an undistorted UAV image", fontsize=13.5, pad=10)
    axA.set_xlabel(
        "cyan = stay_cable; note the polygon covers sky\nand river far beyond the cable strands",
        fontsize=11.5, color="dimgray",
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
    parser = argparse.ArgumentParser(description="Render Figure 3 (stay_cable annotation region + voting)")
    parser.add_argument(
        "--image", default=os.path.join(project_root, "outputs", "undistorted_images", "300.png")
    )
    parser.add_argument(
        "--mask", default=os.path.join(project_root, "outputs", "undistorted_gt_masks", "300.png"),
        help="Class-ID mask in the same undistorted coordinate system as --image",
    )
    parser.add_argument("--crop", nargs=4, type=int, default=[0, 780, 0, 950], metavar=("Y0", "Y1", "X0", "X1"))
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "paper", "figures", "fig3_cable_voting.png"),
    )
    args = parser.parse_args()

    out = build_cable_voting_figure(args.image, args.mask, tuple(args.crop), args.output)
    print(f"[plot_cable_voting_figure] wrote {out}")


if __name__ == "__main__":
    main()
