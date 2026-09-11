"""Draw the compact Task A SegFormer architecture in PNG, PDF and SVG formats."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(17.8, 5.0))
    ax.set(xlim=(0, 18), ylim=(0, 5))
    ax.axis("off")
    ink = "#263c50"

    def box(x, y, w, h, label, color="#eaf1f9"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor=ink, linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=14, color=ink, linespacing=1.5)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=19,
                                    linewidth=1.6, color=ink, shrinkA=3, shrinkB=3))

    box(.25, 3.1, 2.35, 1.0, "RGB image\nResize + normalize")
    ax.text(7.9, 4.65, "MiT-B0 hierarchical encoder", ha="center",
            fontsize=17, fontweight="bold", color=ink)
    xs = [3.4, 5.8, 8.2, 10.6]
    for i, x in enumerate(xs):
        box(x, 3.1, 1.8, 1.0, f"Stage {i + 1}\n1/{2 ** (i + 2)} scale")
        arrow((x + .9, 3.1), (x + .9, 1.95))
        if i:
            arrow((xs[i - 1] + 1.8, 3.6), (x, 3.6))
    arrow((2.6, 3.6), (3.4, 3.6))
    box(3.4, .85, 9.0, 1.1,
        "All-MLP decoder\nChannel projection → resize to 1/4 → concatenate → fuse",
        "#fff0cf")
    ax.text(7.9, .35, "Four feature scales are fused for pixel-level prediction",
            fontsize=13, ha="center", color=ink)
    box(13.4, .85, 4.25, 1.1, "5-class classifier\nLogits at 1/4 input resolution", "#fff0cf")
    arrow((12.4, 1.4), (13.4, 1.4))
    box(13.4, 3.1, 4.25, 1.0,
        "Resize logits → Argmax\nPseudo-mask at original image size", "#eaf5ee")
    arrow((15.525, 1.95), (15.525, 3.1))
    ax.text(15.525, 4.65, "Semantic supervision for Task B", ha="center",
            fontsize=14, fontweight="bold", color=ink)
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98)
    out = Path(__file__).resolve().parents[2] / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fig2_segformer_architecture.{ext}", dpi=250, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
