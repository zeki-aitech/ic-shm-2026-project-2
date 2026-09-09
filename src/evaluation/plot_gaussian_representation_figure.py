"""
Renders Figure 3 (Section 3.4): a left-to-right "model architecture" style diagram of the
semantic Gaussian representation and fused single-pass rasterization - this paper's central
design choice - not tied to one specific real Gaussian or checkpoint, purely illustrative of the
mechanism described in the "Representation" and "Fused rendering" paragraphs.

Drawn as a shared-trunk, dual-head architecture (a familiar convention for multi-task/multi-output
models): every Gaussian's appearance (RGB) and semantic-logit vectors are shown as channel bars
that concatenate into one 8-channel tensor; the geometric parameters bypass this concatenation and
feed the rasterizer directly (they determine projection/depth order, not what gets composited);
the shared `gsplat` rasterization trunk in the middle fans out right into two heads - a rendered
RGB image and a rendered semantic logit map - that are pixel-aligned by construction since both
come from the same trunk.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

STANDARD_EDGE = "#2a4d7a"
STANDARD_FACE = "#e8eef7"
OURS_EDGE = "#b5570f"
OURS_FACE = "#fdeee0"
RGB_CELL_COLORS = ["#e05a4e", "#4caf6d", "#4a7fd6"]
SEM_CELL_COLOR = "#f0b35a"
TRUNK_FACE = "#d8e6f9"
TRUNK_EDGE = "#3a3a3a"
HEAD_RGB_FACE = "#fde8e6"
HEAD_RGB_EDGE = "#a83a2e"
HEAD_SEM_FACE = "#e6f4ea"
HEAD_SEM_EDGE = "#2f6b3d"
ARROW_COLOR = "#333333"


def _box(ax, cx, cy, w, h, text, face, edge, fontsize=12, fontweight="normal"):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.09",
        facecolor=face, edgecolor=edge, linewidth=1.8, zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight, zorder=3)
    return (cx, cy - h / 2), (cx, cy + h / 2), (cx - w / 2, cy), (cx + w / 2, cy)


def _channel_bar(ax, cx, cy, cell_w, cell_h, colors, edge="#333333"):
    """A horizontal register of `len(colors)` colored unit cells - the "tensor as a bar of
    channels" visual used throughout: RGB's 3 cells, the semantic logits' 5 cells, and their
    8-cell concatenation."""
    n = len(colors)
    x0 = cx - n * cell_w / 2
    for i, col in enumerate(colors):
        ax.add_patch(Rectangle((x0 + i * cell_w, cy - cell_h / 2), cell_w, cell_h,
                                facecolor=col, edgecolor=edge, linewidth=1.3, zorder=2))
    return (x0, cy), (x0 + n * cell_w, cy)  # left mid, right mid


def _arrow(ax, p_from, p_to, lw=2.0, mutation_scale=20):
    ax.add_patch(FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=mutation_scale,
        color=ARROW_COLOR, linewidth=lw, zorder=1, shrinkA=2, shrinkB=2,
    ))


def _elbow_arrow(ax, p_from, x_bend, p_to, lw=1.8):
    """Horizontal-vertical-horizontal connector bending at `x_bend` - keeps a long feed clear of
    boxes in between instead of cutting a diagonal line through them."""
    ax.plot([p_from[0], x_bend], [p_from[1], p_from[1]], color=ARROW_COLOR, linewidth=lw, zorder=1)
    ax.plot([x_bend, x_bend], [p_from[1], p_to[1]], color=ARROW_COLOR, linewidth=lw, zorder=1)
    _arrow(ax, (x_bend, p_to[1]), p_to, lw=lw)


def plot_gaussian_representation_figure(output_path: str) -> str:
    fig, ax = plt.subplots(figsize=(17.2, 7.2), dpi=200)
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # ---- Column 1: a single Gaussian and its parameters (left) ----
    ax.text(1.7, 6.75, r"Every Gaussian $g_k$", ha="center", fontsize=13, fontweight="bold")

    geo_b, geo_t, geo_l, geo_r = _box(
        ax, 1.7, 5.55, 3.0, 1.3,
        r"Geometry (3DGS [6])" "\n" r"$\mu_k, s_k, q_k, \alpha_k$",
        face=STANDARD_FACE, edge=STANDARD_EDGE, fontsize=11.5,
    )
    ax.text(1.7, 4.75, "position, scale, rotation, opacity", ha="center", fontsize=9,
            style="italic", color="dimgray")

    ax.text(1.7, 3.55, r"RGB color $c_k$", ha="center", fontsize=11.5, color=STANDARD_EDGE, fontweight="bold")
    rgb_l, rgb_r = _channel_bar(ax, 1.7, 3.0, 0.55, 0.55, RGB_CELL_COLORS, edge=STANDARD_EDGE)

    ax.text(1.7, 1.85, r"Semantic logits $\ell_k$" "\n(this paper's addition)",
            ha="center", fontsize=11.5, color=OURS_EDGE, fontweight="bold")
    sem_l, sem_r = _channel_bar(ax, 1.7, 1.05, 0.42, 0.42, [SEM_CELL_COLOR] * 5, edge=OURS_EDGE)

    # ---- Column 2: concatenation into the 8-channel tensor ----
    cat_x = 4.75
    ax.text(cat_x, 3.85, "concatenate\n" r"$[c_k \,;\, \ell_k] \in \mathbb{R}^8$",
            ha="center", fontsize=11, style="italic", color="dimgray")
    cat_colors = RGB_CELL_COLORS + [SEM_CELL_COLOR] * 5
    cat_l, cat_r = _channel_bar(ax, cat_x, 3.0, 0.42, 0.5, cat_colors, edge="#333333")
    _arrow(ax, rgb_r, (cat_l[0] - 0.15, 3.0 + 0.35))
    _arrow(ax, sem_r, (cat_l[0] - 0.15, 3.0 - 0.35))

    # ---- Column 3: shared rasterization trunk ----
    trunk_cx = 8.9
    trunk_b, trunk_t, trunk_l, trunk_r = _box(
        ax, trunk_cx, 3.0, 3.7, 3.6,
        "gsplat [16]\n\nproject\n↓\ndepth-sort\n↓\nalpha-composite\n\n(shared trunk,\none pass)",
        face=TRUNK_FACE, edge=TRUNK_EDGE, fontsize=12, fontweight="bold",
    )
    _arrow(ax, (cat_r[0] + 0.1, 3.0), (trunk_l[0] - 0.05, 3.0))
    _elbow_arrow(ax, (geo_r[0] + 0.05, geo_r[1]), 6.55, (trunk_l[0] - 0.05, trunk_t[1] - 0.3))

    # ---- Column 4: two output heads, fanning out right ----
    head_x = 14.8
    head_w = 4.4
    head_l_edge = head_x - head_w / 2  # 12.6

    rgb_head_b, rgb_head_t, rgb_head_l, rgb_head_r = _box(
        ax, head_x, 5.15, head_w, 1.65,
        "RGB head\nRendered RGB image (3 ch.)",
        face=HEAD_RGB_FACE, edge=HEAD_RGB_EDGE, fontsize=12, fontweight="bold",
    )
    sem_head_b, sem_head_t, sem_head_l, sem_head_r = _box(
        ax, head_x, 0.85, head_w, 1.65,
        "Semantic head\nRendered logit map (5 ch.)\n→ arg max → class map",
        face=HEAD_SEM_FACE, edge=HEAD_SEM_EDGE, fontsize=12, fontweight="bold",
    )

    bend_x = (trunk_r[0] + head_l_edge) / 2  # midpoint of the trunk-to-head gap
    _elbow_arrow(ax, (trunk_r[0] + 0.05, trunk_t[1] - 0.5), bend_x, (head_l_edge - 0.05, 5.15))
    _elbow_arrow(ax, (trunk_r[0] + 0.05, trunk_b[1] + 0.5), bend_x, (head_l_edge - 0.05, 0.85))

    ax.annotate("", xy=(head_l_edge - 0.7, 3.5), xytext=(head_l_edge - 0.7, 2.5),
                arrowprops=dict(arrowstyle="<->", color="dimgray", linewidth=1.4))
    ax.text(head_l_edge - 0.5, 3.0, "pixel-aligned\nby construction", ha="left", va="center",
            fontsize=9.5, style="italic", color="dimgray")

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
