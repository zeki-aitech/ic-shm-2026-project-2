"""
Assembles Figure 5 (Section 5.3): a filmstrip of RGB + semantic frames along a camera path
interpolated between two real flown poses (`src.gaussian_splatting.interpolate`), demonstrating
rendering from genuinely arbitrary viewpoints with pixel-aligned RGB/semantic outputs.
"""
import glob
import os
import re
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.evaluation.plot_qualitative_grid import colorize_mask

_FRAME_RE = re.compile(r"^t(\d+)_([0-9.]+)_rgb\.png$")


def _discover_frames(frame_dir: str) -> List[Tuple[int, float, str]]:
    """Finds `t{idx}_{t}_rgb.png` files in `frame_dir` and returns `(idx, t, stem)` sorted by idx.
    `stem` is `t{idx}_{t}` (i.e. the shared prefix before `_rgb.png`/`_sem.png`)."""
    frames = []
    for path in glob.glob(os.path.join(frame_dir, "t*_rgb.png")):
        m = _FRAME_RE.match(os.path.basename(path))
        if m:
            idx, t = int(m.group(1)), float(m.group(2))
            stem = f"t{m.group(1)}_{m.group(2)}"
            frames.append((idx, t, stem))
    frames.sort(key=lambda x: x[0])
    return frames


def build_interpolation_filmstrip(frame_dir: str, output_path: str) -> str:
    frames = _discover_frames(frame_dir)
    if not frames:
        raise FileNotFoundError(f"No t*_rgb.png frames found in {frame_dir}")

    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.6, 2 * 2.0), dpi=200)
    if n == 1:
        axes = axes[:, None]

    for col, (idx, t, stem) in enumerate(frames):
        rgb = np.asarray(Image.open(os.path.join(frame_dir, f"{stem}_rgb.png")).convert("RGB"))
        sem = colorize_mask(np.asarray(Image.open(os.path.join(frame_dir, f"{stem}_sem.png"))))

        for row, img in enumerate((rgb, sem)):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[0, col].set_title(f"t={t:.2f}", fontsize=15, fontweight="bold")

    axes[0, 0].set_ylabel("RGB", fontsize=14, fontweight="bold", rotation=90, va="center")
    axes[1, 0].set_ylabel("Semantic", fontsize=14, fontweight="bold", rotation=90, va="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description="Assemble Figure 5 (interpolation filmstrip)")
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "paper", "figures", "fig7_interpolation.png"),
    )
    args = parser.parse_args()

    out = build_interpolation_filmstrip(args.frame_dir, args.output)
    print(f"[plot_interpolation_sequence] wrote {out}")


if __name__ == "__main__":
    main()
