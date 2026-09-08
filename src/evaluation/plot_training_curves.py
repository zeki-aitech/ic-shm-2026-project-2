"""
Renders Figures 7-8 (Section 5.5): training convergence curves for Task A (SegFormer
fine-tuning) and Task B (semantic Gaussian Splatting), parsed directly from the real training
logs saved to `outputs/logs/` during the actual runs used elsewhere in the paper - not
regenerated or simulated.
"""
import os
import re
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TASK_A_RE = re.compile(r"epoch (\d+)/\d+ loss=([\d.]+) val_mIoU=([\d.]+)")
_TASK_B_RE = re.compile(r"step (\d+)/\d+ loss=([\d.]+) n_gaussians=(\d+)")


def parse_segmentation_log(log_path: str) -> Tuple[List[int], List[float], List[float]]:
    """Returns (epochs, train_loss, val_mIoU) parsed from a `[segmentation] epoch e/E
    loss=... val_mIoU=...` training log."""
    epochs, losses, mious = [], [], []
    with open(log_path) as f:
        for line in f:
            m = _TASK_A_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                mious.append(float(m.group(3)))
    return epochs, losses, mious


def parse_gaussian_log(log_path: str) -> Tuple[List[int], List[float]]:
    """Returns (steps, loss) parsed from a `[gaussian_splatting] step s/S loss=...
    n_gaussians=...` training log."""
    steps, losses = [], []
    with open(log_path) as f:
        for line in f:
            m = _TASK_B_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses


def _smooth(values: List[float], window: int = 15) -> List[float]:
    """Trailing moving average, used only as a visual overlay on top of the noisy raw
    per-step loss (each step's loss is one mini-batch of rendered pixels, not an epoch
    average, so it fluctuates far more than Task A's per-epoch loss)."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i + 1 - lo))
    return out


def plot_task_a_training_curve(log_path: str, output_path: str) -> str:
    epochs, losses, mious = parse_segmentation_log(log_path)

    fig, ax1 = plt.subplots(figsize=(7.5, 5), dpi=200)
    color_loss, color_miou = "#c0392b", "#2166ac"

    ax1.plot(epochs, losses, color=color_loss, linewidth=2.2)
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Training loss", color=color_loss, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_title("Task A: SegFormer fine-tuning convergence", fontsize=14.5, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [m * 100 for m in mious], color=color_miou, linewidth=2.2)
    ax2.set_ylabel("Validation mIoU (%)", color=color_miou, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=color_miou)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_task_b_training_curve(log_path: str, output_path: str) -> str:
    steps, losses = parse_gaussian_log(log_path)
    smoothed = _smooth(losses)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    ax.plot(steps, losses, color="#b0b0b0", linewidth=1.0, label="raw (per 100 steps)")
    ax.plot(steps, smoothed, color="#1a7a3d", linewidth=2.4, label="moving average")
    ax.set_xlabel("Training step", fontsize=13)
    ax.set_ylabel("Training loss", fontsize=13)
    ax.set_title("Task B: Semantic Gaussian Splatting convergence", fontsize=14.5, fontweight="bold")
    ax.legend(fontsize=11.5, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description="Render Task A / Task B training curve figures")
    parser.add_argument("--task", choices=["a", "b"], required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    default_name = "fig7_task_a_training.png" if args.task == "a" else "fig8_task_b_training.png"
    output = args.output or os.path.join(project_root, "paper", "figures", default_name)
    fn = plot_task_a_training_curve if args.task == "a" else plot_task_b_training_curve
    out = fn(args.log, output)
    print(f"[plot_training_curves] wrote {out}")


if __name__ == "__main__":
    main()
