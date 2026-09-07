"""
Renders Figure 1 (Section 3): the two-branch pipeline overview diagram. Purely illustrative of
the fixed architecture (Task A / Task B / rendering) - unlike Figures 3-5, it does not visualize
data from a specific run, so it takes no arguments beyond the output path.
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


def _box(ax, cx, cy, w, h, text, face=STAGE_COLOR, edge=STAGE_EDGE, fontsize=9.5, fontweight="normal"):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=face, edgecolor=edge, linewidth=1.4, zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight, zorder=3)
    return (cx, cy - h / 2), (cx, cy + h / 2), (cx - w / 2, cy), (cx + w / 2, cy)  # bottom,top,left,right


def _arrow(ax, p_from, p_to, label=None, label_offset=(0.15, 0)):
    arrow = FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=14,
        color=ARROW_COLOR, linewidth=1.3, zorder=1, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label, ha="left", va="center",
                fontsize=8, style="italic", color="dimgray", zorder=3)


def plot_pipeline_diagram(output_path: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 11), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 15)
    ax.axis("off")

    left_x, right_x = 2.8, 7.2
    bw, bh = 4.6, 1.3

    # Row 0: input
    in_b, in_t, in_l, in_r = _box(
        ax, 5, 14, 7.0, 1.1,
        "400 UAV images (300 labeled + 100 unlabeled)\n+ COLMAP camera poses",
        fontweight="bold",
    )

    # Row 1
    a1_b, a1_t, a1_l, a1_r = _box(ax, left_x, 12.1, bw, bh, "COLMAP triangulation\n→ sparse point cloud\n(84,613 points)")
    b1_b, b1_t, b1_l, b1_r = _box(ax, right_x, 12.1, bw, bh, "Task A: fine-tune SegFormer\n(240 labeled training views)")

    # Row 2
    a2_b, a2_t, a2_l, a2_r = _box(ax, left_x, 10.2, bw, bh, "Multi-view semantic voting\n(strict-majority rule for cable)")
    b2_b, b2_t, b2_l, b2_r = _box(ax, right_x, 10.2, bw, bh, "Predict pseudo-masks for\n100 unlabeled images")

    # Row 3
    a3_b, a3_t, a3_l, a3_r = _box(ax, left_x, 8.3, bw, bh, "Semantic warm-start\n(Gaussian means, colors, logits)")

    # Merge box
    merge_b, merge_t, merge_l, merge_r = _box(
        ax, 5, 6.1, 8.4, 1.5,
        "Task B: Semantic 3D Gaussian Splatting training\nfused single-pass RGB + semantic rasterization",
        face=MERGE_COLOR, edge=MERGE_EDGE, fontweight="bold",
    )

    # Trained model
    tm_b, tm_t, tm_l, tm_r = _box(ax, 5, 4.2, 5.0, 1.1, "Trained model\n(602,363 Gaussians)")

    # Render
    r_b, r_t, r_l, r_r = _box(ax, 5, 2.4, 6.5, 1.0, "render(pose): arbitrary camera viewpoint", fontweight="bold")

    # Outputs
    o1_b, o1_t, o1_l, o1_r = _box(ax, left_x, 0.4, bw - 0.6, 1.0, "RGB image", face=OUTPUT_COLOR, edge=OUTPUT_EDGE)
    o2_b, o2_t, o2_l, o2_r = _box(ax, right_x, 0.4, bw - 0.6, 1.0, "Semantic map\n(classes 0–4)", face=OUTPUT_COLOR, edge=OUTPUT_EDGE)

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


def main():
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description="Render Figure 1 (pipeline overview diagram)")
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "paper", "figures", "fig1_pipeline.png"),
    )
    args = parser.parse_args()
    out = plot_pipeline_diagram(args.output)
    print(f"[plot_pipeline_diagram] wrote {out}")


if __name__ == "__main__":
    main()
