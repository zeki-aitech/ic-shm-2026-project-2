"""
Renders Figure 1 (Section 3): the two-branch pipeline overview diagram. Purely illustrative of
the fixed architecture (Task A / Task B / rendering) - unlike Figures 3-5, it does not visualize
data from a specific run, so it takes no arguments beyond the output path.

`plot_pipeline_diagram_horizontal` (left-to-right: input/preprocessing/Task A on the left, Task B
onward on the right) is the version actually used for the paper figure. `plot_pipeline_diagram`
(top-to-bottom) is kept as an alternative layout.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STAGE_COLOR = "#e8eef7"
STAGE_EDGE = "#2a4d7a"
MERGE_COLOR = "#fdf1d6"
MERGE_EDGE = "#8a6d1a"
OUTPUT_COLOR = "#e6f4ea"
OUTPUT_EDGE = "#2f6b3d"
ARROW_COLOR = "#333333"


def _box(ax, cx, cy, w, h, text, face=STAGE_COLOR, edge=STAGE_EDGE, fontsize=15, fontweight="normal"):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=face, edgecolor=edge, linewidth=1.8, zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight, zorder=3)
    return (cx, cy - h / 2), (cx, cy + h / 2), (cx - w / 2, cy), (cx + w / 2, cy)  # bottom,top,left,right


def _arrow(ax, p_from, p_to, label=None, label_offset=(0.15, 0)):
    arrow = FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=18,
        color=ARROW_COLOR, linewidth=1.6, zorder=1, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label, ha="left", va="center",
                fontsize=12.5, style="italic", color="dimgray", zorder=3)


def plot_pipeline_diagram(output_path: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 15), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.0, 17.5)
    ax.axis("off")

    left_x, right_x = 2.8, 7.2
    bw, bh = 4.8, 1.8

    # Row 0: input
    in_b, in_t, in_l, in_r = _box(
        ax, 5, 16.3, 7.4, 1.4,
        "400 UAV images (300 labeled + 100 unlabeled)\n+ COLMAP camera poses",
        fontsize=16, fontweight="bold",
    )

    # Row 1
    a1_b, a1_t, a1_l, a1_r = _box(ax, left_x, 14.0, bw, bh, "COLMAP triangulation\n→ sparse point cloud\n(84,613 points)")
    b1_b, b1_t, b1_l, b1_r = _box(ax, right_x, 14.0, bw, bh, "Task A: fine-tune SegFormer\n(240 labeled training views)")

    # Row 2
    a2_b, a2_t, a2_l, a2_r = _box(ax, left_x, 11.6, bw, bh, "Multi-view semantic voting\n(strict-majority rule for cable)")
    b2_b, b2_t, b2_l, b2_r = _box(ax, right_x, 11.6, bw, bh, "Predict pseudo-masks for\n100 unlabeled images")

    # Row 3
    a3_b, a3_t, a3_l, a3_r = _box(ax, left_x, 9.2, bw, bh, "Semantic warm-start\n(Gaussian means, colors, logits)")

    # Merge box
    merge_b, merge_t, merge_l, merge_r = _box(
        ax, 5, 6.7, 8.8, 1.9,
        "Task B: Semantic 3D Gaussian Splatting training\nfused single-pass RGB + semantic rasterization",
        face=MERGE_COLOR, edge=MERGE_EDGE, fontsize=16, fontweight="bold",
    )

    # Trained model
    tm_b, tm_t, tm_l, tm_r = _box(ax, 5, 4.5, 5.4, 1.4, "Trained model\n(602,363 Gaussians)")

    # Render
    r_b, r_t, r_l, r_r = _box(ax, 5, 2.4, 7.0, 1.3, "render(pose): arbitrary camera viewpoint",
                               fontsize=16, fontweight="bold")

    # Outputs
    o1_b, o1_t, o1_l, o1_r = _box(ax, left_x, 0.2, bw - 0.6, 1.3, "RGB image", face=OUTPUT_COLOR, edge=OUTPUT_EDGE)
    o2_b, o2_t, o2_l, o2_r = _box(ax, right_x, 0.2, bw - 0.6, 1.3, "Semantic map\n(classes 0–4)", face=OUTPUT_COLOR, edge=OUTPUT_EDGE)

    # Arrows: input -> two branches
    _arrow(ax, (5 - 0.1, in_b[1]), (left_x, a1_t[1] + 0.05))
    _arrow(ax, (5 + 0.1, in_b[1]), (right_x, b1_t[1] + 0.05))

    # Left branch chain
    _arrow(ax, a1_b, a2_t)
    _arrow(ax, a2_b, a3_t)

    # Right branch chain
    _arrow(ax, b1_b, b2_t)

    # Both branches -> merge
    _arrow(ax, a3_b, (left_x, merge_t[1] + 0.05), label="warm-start")
    _arrow(ax, (right_x, b2_b[1]), (right_x, merge_t[1] + 0.05), label="340 supervised views")

    _arrow(ax, merge_b, tm_t)
    _arrow(ax, tm_b, r_t)
    _arrow(ax, (5 - 0.1, r_b[1]), (left_x, o1_t[1] + 0.05))
    _arrow(ax, (5 + 0.1, r_b[1]), (right_x, o2_t[1] + 0.05))

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pipeline_diagram_horizontal(output_path: str) -> str:
    """Same content as `plot_pipeline_diagram`, laid out left-to-right instead of top-to-bottom:
    input + preprocessing + Task A on the left, Task B training through to the rendered outputs
    on the right. Task B's box is deliberately sized so its top-left and bottom-left corners
    land exactly on the y-coordinates of the two left-column source boxes (pseudo-masks and
    warm-start respectively), so both cross-over connectors are perfectly horizontal straight
    lines instead of diagonals or elbows - no crossings, no arrows cutting through boxes."""
    fig, ax = plt.subplots(figsize=(19, 9.2), dpi=200)
    ax.set_xlim(0, 19)
    ax.set_ylim(-1.0, 12.6)
    ax.axis("off")

    sub_l, sub_r = 2.3, 6.9
    bw, bh = 4.0, 1.7
    row1_y, row2_y, row3_y = 9.3, 6.9, 4.5

    # Left column: input
    in_b, in_t, in_l, in_r = _box(
        ax, 4.6, 11.4, 8.8, 1.6,
        "400 UAV images (300 labeled + 100 unlabeled)\n+ COLMAP camera poses",
        fontsize=14.5, fontweight="bold",
    )

    # Left column, two sub-branch chains. Task A and Task B share a color (MERGE_COLOR) to mark
    # them visually as the pipeline's two named tasks, distinct from the plain preprocessing
    # steps (COLMAP/voting/warm-start/pseudo-labeling) around them.
    a1_b, a1_t, a1_l, a1_r = _box(ax, sub_l, row1_y, bw, bh, "COLMAP triangulation\n→ sparse point cloud\n(84,613 points)", fontsize=13.5)
    b1_b, b1_t, b1_l, b1_r = _box(ax, sub_r, row1_y, bw, bh, "Task A: fine-tune SegFormer\n(240 labeled training views)",
                                   face=MERGE_COLOR, edge=MERGE_EDGE, fontsize=13.5, fontweight="bold")

    a2_b, a2_t, a2_l, a2_r = _box(ax, sub_l, row2_y, bw, bh, "Multi-view semantic voting\n(strict-majority rule for cable)", fontsize=13.5)
    b2_b, b2_t, b2_l, b2_r = _box(ax, sub_r, row2_y, bw, bh, "Predict pseudo-masks for\n100 unlabeled images", fontsize=13.5)

    a3_b, a3_t, a3_l, a3_r = _box(ax, sub_l, row3_y, bw, bh, "Semantic warm-start\n(Gaussian means,\ncolors, logits)", fontsize=13.5)

    # Right column: Task B's top is level with row 1 (Task A's row) and its bottom lands exactly
    # on row3_y, so it is visually paired with Task A (same top, same color) while both
    # cross-over connectors below still land on its left edge and run perfectly horizontal.
    right_x = 14.3
    row1_top = row1_y + bh / 2
    merge_h = row1_top - row3_y
    merge_cy = (row1_top + row3_y) / 2
    merge_b, merge_t, merge_l, merge_r = _box(
        ax, right_x, merge_cy, 8.6, merge_h,
        "Task B: Semantic 3D Gaussian Splatting training\nfused single-pass RGB + semantic rasterization",
        face=MERGE_COLOR, edge=MERGE_EDGE, fontsize=15, fontweight="bold",
    )
    tm_b, tm_t, tm_l, tm_r = _box(ax, right_x, merge_b[1] - 1.05, 6.0, 1.4, "Trained model\n(602,363 Gaussians)", fontsize=15)
    r_b, r_t, r_l, r_r = _box(ax, right_x, tm_b[1] - 0.95, 7.4, 1.3, "render(pose): arbitrary camera viewpoint",
                               fontsize=15, fontweight="bold")
    out_cy = r_b[1] - 0.95
    o1_b, o1_t, o1_l, o1_r = _box(ax, right_x - 2.1, out_cy, 3.4, 1.3, "RGB image", face=OUTPUT_COLOR, edge=OUTPUT_EDGE, fontsize=14)
    o2_b, o2_t, o2_l, o2_r = _box(ax, right_x + 2.1, out_cy, 3.4, 1.3, "Semantic map\n(classes 0–4)", face=OUTPUT_COLOR, edge=OUTPUT_EDGE, fontsize=14)

    # Arrows within left column
    _arrow(ax, (4.6 - 0.1, in_b[1]), (sub_l, a1_t[1] + 0.05))
    _arrow(ax, (4.6 + 0.1, in_b[1]), (sub_r, b1_t[1] + 0.05))
    _arrow(ax, a1_b, a2_t)
    _arrow(ax, a2_b, a3_t)
    _arrow(ax, b1_b, b2_t)

    # Cross-over connectors: perfectly horizontal by construction (see merge_h/merge_cy above).
    # The pseudo-mask connector is short (only 1.1 units), so its label is placed in the clear
    # gap between rows 1 and 2 instead of on the line itself, where it would collide with the
    # "Predict pseudo-masks" box; the warm-start connector is long enough for a directly
    # above-the-line label.
    _arrow(ax, b2_r, (merge_l[0], row2_y))
    ax.text(b2_r[0], (row1_y - bh / 2 + row2_y + bh / 2) / 2, "340 supervised views",
            ha="right", va="center", fontsize=12.5, style="italic", color="dimgray", zorder=3)
    _arrow(ax, a3_r, (merge_l[0], row3_y), label="warm-start", label_offset=(-0.55, 0.22))

    # Right column chain
    for p_from, p_to in [
        (merge_b, tm_t),
        (tm_b, r_t),
        ((right_x - 0.1, r_b[1]), (right_x - 2.1, o1_t[1] + 0.05)),
        ((right_x + 0.1, r_b[1]), (right_x + 2.1, o2_t[1] + 0.05)),
    ]:
        _arrow(ax, p_from, p_to)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description="Render Figure 1 (pipeline overview diagram)")
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "paper", "figures", "fig1_pipeline.png"),
    )
    parser.add_argument("--horizontal", action="store_true", help="Left-to-right layout instead of top-to-bottom")
    args = parser.parse_args()
    out = (plot_pipeline_diagram_horizontal if args.horizontal else plot_pipeline_diagram)(args.output)
    print(f"[plot_pipeline_diagram] wrote {out}")


if __name__ == "__main__":
    main()
