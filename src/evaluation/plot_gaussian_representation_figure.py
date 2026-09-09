"""
Renders Figure 3 (Section 3.4): a schematic of the semantic Gaussian representation and the
fused single-pass RGB+semantic rasterization that is this paper's central design choice - not
tied to one specific real Gaussian or checkpoint, purely illustrative of the mechanism described
in the "Representation" and "Fused rendering" paragraphs.

Shows: (1) a single Gaussian's parameters, split into the standard 3DGS set (mean, scale,
rotation, opacity - which determine projection and depth order) and this paper's addition (a
per-Gaussian semantic logit vector, concatenated with RGB color into the 8-channel tensor that is
actually composited), (2) many such Gaussians projected, depth-sorted, and alpha-composited in a
single rasterization pass with shared weights, and (3) that one pass's output split into a
rendered RGB image and a rendered semantic logit map (resolved to a class map by arg max at
read-out time).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STANDARD_COLOR = "#e8eef7"
STANDARD_EDGE = "#2a4d7a"
OURS_COLOR = "#fdeee0"
OURS_EDGE = "#b5570f"
MERGE_COLOR = "#fdf1d6"
MERGE_EDGE = "#8a6d1a"
OUTPUT_COLOR = "#e6f4ea"
OUTPUT_EDGE = "#2f6b3d"
GAUSSIAN_COLOR = "#d8e6f9"
GAUSSIAN_EDGE = "#3a3a3a"
ARROW_COLOR = "#333333"


def _box(ax, cx, cy, w, h, text, face=STANDARD_COLOR, edge=STANDARD_EDGE, fontsize=13, fontweight="normal"):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.07,rounding_size=0.1",
        facecolor=face, edgecolor=edge, linewidth=1.8, zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight, zorder=3)
    return (cx, cy - h / 2), (cx, cy + h / 2), (cx - w / 2, cy), (cx + w / 2, cy)


def _arrow(ax, p_from, p_to):
    arrow = FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=22,
        color=ARROW_COLOR, linewidth=2.0, zorder=1, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)


def _zigzag_arrow(ax, p_from, x_bend, p_to, label=None, fontsize=10.5):
    """Horizontal-vertical-horizontal connector bending at `x_bend`, so a long feed (e.g. the
    geometric params reaching all the way down to the rasterization box) stays clear of the
    boxes in between instead of cutting a diagonal line through them."""
    ax.plot([p_from[0], x_bend], [p_from[1], p_from[1]], color=ARROW_COLOR, linewidth=2.0, zorder=1)
    ax.plot([x_bend, x_bend], [p_from[1], p_to[1]], color=ARROW_COLOR, linewidth=2.0, zorder=1)
    _arrow(ax, (x_bend, p_to[1]), p_to)
    if label:
        mid_y = (p_from[1] + p_to[1]) / 2
        ax.text(x_bend - 0.15, mid_y, label, rotation=90, ha="right", va="center",
                fontsize=fontsize, style="italic", color="dimgray", zorder=3)


def plot_gaussian_representation_figure(output_path: str) -> str:
    fig, ax = plt.subplots(figsize=(13, 9.7), dpi=200)
    ax.set_xlim(0, 13)
    ax.set_ylim(0.5, 11)
    ax.axis("off")

    # Row 1: per-Gaussian parameters, split standard vs. this paper's addition.
    ax.text(3.6, 10.55, "Standard 3D Gaussian Splatting [6]", ha="center", fontsize=12.5,
             fontweight="bold", color=STANDARD_EDGE)
    std_b, std_t, std_l, std_r = _box(
        ax, 3.6, 9.5, 6.0, 1.5,
        r"$\mu_k \in \mathbb{R}^3$   $s_k \in \mathbb{R}^3$   $q_k \in \mathbb{R}^4$   $\alpha_k \in \mathbb{R}$"
        "\n(mean, scale, rotation, opacity)",
        fontsize=12.5,
    )
    c_b, c_t, c_l, c_r = _box(ax, 3.6, 7.7, 3.2, 1.1, r"RGB color $c_k \in \mathbb{R}^3$", fontsize=13)

    ax.text(10.3, 10.55, "This paper's addition", ha="center", fontsize=12.5,
             fontweight="bold", color=OURS_EDGE)
    ell_b, ell_t, ell_l, ell_r = _box(
        ax, 10.3, 9.15, 4.6, 1.9,
        r"Semantic logit vector"
        "\n" r"$\ell_k \in \mathbb{R}^5$"
        "\n(one entry per structural class)",
        face=OURS_COLOR, edge=OURS_EDGE, fontsize=12.5,
    )

    ax.text(6.95, 8.45, "every\nGaussian $g_k$", ha="center", va="center", fontsize=11,
            style="italic", color="dimgray")

    # Concatenation into the 8-channel tensor rasterized as one.
    cat_b, cat_t, cat_l, cat_r = _box(
        ax, 6.95, 6.0, 6.4, 1.2,
        r"Concatenate: $[c_k \,;\, \ell_k] \in \mathbb{R}^8$ — one 8-channel tensor per Gaussian",
        face=MERGE_COLOR, edge=MERGE_EDGE, fontweight="bold", fontsize=13,
    )
    _arrow(ax, (c_r[0] + 0.05, c_r[1]), (cat_l[0] - 1.6, cat_t[1] + 0.5))
    _arrow(ax, (ell_l[0] - 0.05, ell_b[1] - 0.05), (cat_l[0] + 1.6, cat_t[1] + 0.5))

    # Single fused rasterization pass.
    ras_b, ras_t, ras_l, ras_r = _box(
        ax, 6.95, 4.05, 10.4, 1.4,
        "gsplat [16]: project → depth-sort → alpha-composite\n"
        "(single pass, shared weights for every channel)",
        face=GAUSSIAN_COLOR, edge=GAUSSIAN_EDGE, fontweight="bold", fontsize=13,
    )
    _arrow(ax, cat_b, ras_t)

    # Geometric params bypass the concatenation and feed the rasterizer directly - they
    # determine projection/depth order, not which channels get composited.
    _zigzag_arrow(ax, (std_l[0] - 0.05, std_l[1]), 0.9, (ras_l[0] + 0.15, ras_t[1] - 0.15),
                  label="position, scale, rotation, opacity\n(determine projection & depth order)")

    # Split outputs, pixel-aligned by construction (explained in the external figure caption).
    o1_b, o1_t, o1_l, o1_r = _box(ax, 4.15, 1.85, 3.6, 1.1, "Rendered RGB image\n(3 channels)",
                                   face=OUTPUT_COLOR, edge=OUTPUT_EDGE, fontsize=12)
    o2_b, o2_t, o2_l, o2_r = _box(ax, 9.55, 1.85, 5.6, 1.1,
                                   "Rendered semantic logit map (5 ch.)\n→ arg max → class map",
                                   face=OUTPUT_COLOR, edge=OUTPUT_EDGE, fontsize=12)
    _arrow(ax, (ras_b[0] - 0.15, ras_b[1]), (4.15, o1_t[1] + 0.05))
    _arrow(ax, (ras_b[0] + 0.15, ras_b[1]), (9.55, o2_t[1] + 0.05))

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    plot_gaussian_representation_figure(
        os.path.join(PROJECT_ROOT, "paper", "figures", "fig3_gaussian_representation.png")
    )
