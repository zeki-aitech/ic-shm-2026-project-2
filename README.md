# IC-SHM 2026 — Project 2: Multi-view Semantic 3D Reconstruction of Bridge Structures

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![gsplat](https://img.shields.io/badge/gsplat-1.5.3-orange.svg)](https://github.com/nerfstudio-project/gsplat)

## 📌 1. Project Overview

This project targets the **official IC-SHM 2026 Project 2 brief** — see
[`data/Contest Dataset/The 4th International Project Competition for SHM_2026.pdf`](<data/Contest Dataset/The 4th International Project Competition for SHM_2026.pdf>),
pp. 9-10:

> Participants must develop a model that reconstructs a semantically labeled 3D representation
> of a bridge structure from multi-view images. Submitted models are evaluated on a blind test
> set of camera viewpoints on two criteria: **Visual Fidelity** (rendered RGB vs. ground-truth
> photos, via PSNR/SSIM/LPIPS) and **Semantic Accuracy** (rendered semantic map vs. ground-truth
> labels, via mIoU). `Accuracy Score = 0.50 × Visual Fidelity + 0.50 × Semantic mIoU`. The
> submitted model/script must render **both** an RGB image and a semantic map (official class
> IDs) from an arbitrary given camera viewpoint.

The pipeline is a **Semantic 3D Gaussian Splatting** model (`src/gaussian_splatting/`): each
Gaussian carries a per-class semantic logit vector alongside the usual geometry/color parameters,
rendered through a single fused rasterization pass so one model produces both the RGB image and
the semantic map the contest scores. A 2D segmentation model (`src/segmentation/`) pseudo-labels
the 100 unlabeled images to widen semantic supervision. Camera pose loading, sparse triangulation,
and 2D→3D semantic voting (`src/colmap_io/`) supply the posed training views and the semantic
warm-start for the Gaussian model's parameters.

---

## 🏗️ 2. Semantic Class Taxonomy

| Class ID | Class Name (`label`) | RGB Color | Hex Color | Description |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `background` | `(128, 128, 128)` | `#808080` | Sky, river, terrain, surroundings |
| **1** | `deck` | `(255, 0, 0)` | `#FF0000` | Bridge deck / roadway girder |
| **2** | `stay_cable` | `(0, 255, 255)` | `#00FFFF` | Cable stay bundles |
| **3** | `tower` | `(0, 255, 0)` | `#00FF00` | Pylons / tower shafts |
| **4** | `foundation` | `(255, 255, 0)` | `#FFFF00` | Bridge piers / substructure footings |

---

## 📂 3. Directory Structure

```text
ic-shm-2026-project-2/
├── docs/                          # Technical specifications and contest documentation
├── src/
│   ├── colmap_io/                  # Camera/pose loading, triangulation, semantic voting
│   │   ├── models.py                  # CameraIntrinsics, ImagePose, Point3D dataclasses
│   │   ├── reconstructor.py           # PycolmapReconstructor: LO-RANSAC sparse triangulation
│   │   └── semantic_voting.py         # SemanticProjector: 2D->3D multi-view majority voting
│   ├── segmentation/               # Task A: 2D pseudo-labeling for unlabeled frames
│   │   ├── dataset.py                 # BridgeSegDataset
│   │   ├── train.py                   # Fine-tunes SegFormer (mit-b0) with 2D mIoU validation
│   │   └── infer.py                   # Predicts pseudo-masks for the 100 unlabeled images
│   ├── gaussian_splatting/         # Task B: Semantic 3D Gaussian Splatting (scored deliverable)
│   │   ├── undistort.py               # One-time lens-undistortion pass (pinhole convention)
│   │   ├── dataset.py                 # GSCamera / build_camera_list
│   │   ├── model.py                   # SemanticGaussianModel (fused RGB + semantic rasterization)
│   │   ├── losses.py                  # Photometric (L1 + D-SSIM) + semantic cross-entropy
│   │   ├── train.py                   # Training loop (gsplat DefaultStrategy densification)
│   │   └── render.py                  # Contest deliverable: render RGB+semantic from any pose
│   ├── evaluation/
│   │   ├── metrics.py                  # Confusion matrix / IoU / mIoU + trajectory_interleaved_split
│   │   └── render_metrics.py           # PSNR/SSIM/LPIPS + mIoU on rendered holdout views
│   └── utils/                      # 2D data preprocessing
│       ├── json_to_mask.py            # Labelme JSON -> uint8 PNG masks (canonical CLASS_MAPPING)
│       └── create_overlay_dataset.py
└── tests/                          # Automated unit tests (uv run pytest)
```

---

## ⚙️ 4. Pipeline Architecture

**Task A — 2D pseudo-labeling** (`src/segmentation/`): fine-tunes SegFormer (mit-b0 backbone) on
the trajectory-interleaved 240-image train split, validates 2D mIoU on the 60-image holdout, then
predicts pseudo-masks for the 100 unlabeled images. The 60 holdout images/masks are never used to
train anything — reserved for the final render-based evaluation.

**Task B — Semantic 3D Gaussian Splatting** (`src/gaussian_splatting/`):
1. **Undistort** (`undistort.py`) all 400 images once to a pinhole convention (`gsplat` renders
   an ideal pinhole camera; the shared COLMAP `SIMPLE_RADIAL` camera has a small but non-zero
   `k1`).
2. **Initialize** Gaussian means/colors from `PycolmapReconstructor`'s triangulated sparse cloud
   (`src/colmap_io/reconstructor.py`), and semantic logits from `SemanticProjector`'s per-point
   voted class (`src/colmap_io/semantic_voting.py`, train-views only).
3. **Train** (`train.py`) on the 240 GT-mask + 100 pseudo-mask = 340 views with photometric
   (L1 + D-SSIM) + semantic cross-entropy loss, using `gsplat.strategy.DefaultStrategy` for
   gradient-driven densification/pruning.
4. **Render** (`render.py`) an RGB image + semantic map (official class IDs) from any camera
   pose — this is the literal contest submission artifact.
5. **Evaluate** (`src/evaluation/render_metrics.py`) on the 60 never-trained holdout views:
   PSNR/SSIM/LPIPS (visual fidelity) + mIoU (semantic accuracy).

---

## 🚀 5. Quickstart Guide

### Environment Setup
This project uses `uv`. `gsplat`'s CUDA kernels are pinned to a prebuilt wheel
(`torch==2.4.1+cu121` / `gsplat==1.5.3+pt24cu121`, see `[tool.uv.sources]` in `pyproject.toml`)
to avoid JIT-compiling CUDA extensions, which is fragile across host CUDA toolkit versions.

```bash
uv sync --extra deeplearning
```

### Running Unit Tests
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```
(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids picking up unrelated pytest plugins from other
system-wide Python environments, e.g. ROS's `launch_testing`.)

### End-to-End Pipeline

```bash
# 1. Ground-truth masks from Labelme JSON (already generated under outputs/gt_masks/, re-run if needed)
uv run python -m src.utils.json_to_mask

# 2. Task A: fine-tune the 2D segmentation model, then pseudo-label the 100 unlabeled images
uv run python -m src.segmentation.train --epochs 80
uv run python -m src.segmentation.infer --checkpoint outputs/checkpoints/segformer_mitb0/best.pt

# 3. Task B: train the Semantic 3D Gaussian Splatting model
#    (auto-undistorts images and computes the semantic warm-start on first run)
uv run python -m src.gaussian_splatting.train --iters 40000 --downsample 1.0

# 4. Contest deliverable: render RGB + semantic map from an arbitrary camera pose
uv run python -m src.gaussian_splatting.render \
    --checkpoint outputs/checkpoints/gaussians/final.pt \
    --pose-line "$(sed -n '5p' 'data/Contest Dataset/camera_parameters/images.txt')" \
    --out-rgb rgb.png --out-sem sem.png

# 5. Evaluate on the 60 held-out (never trained on) views
uv run python -m src.evaluation.render_metrics \
    --checkpoint outputs/checkpoints/gaussians/final.pt \
    --output outputs/eval/render_eval_report.md
```

---

## 📊 6. Results (RTX 3080, 10GB)

Trained at full resolution (1320x989), 40,000 iterations, 84,613 -> 602,363 Gaussians.
Evaluated on the 60-image holdout, never used in training:

| Metric | Value |
| :--- | :---: |
| Task A val 2D mIoU (60-image holdout) | 81.27% |
| PSNR | 22.18 dB |
| SSIM | 0.849 |
| LPIPS | 0.334 |
| **Semantic mIoU (structural, 4 classes)** | **91.47%** |
| Accuracy Score (illustrative) | 0.816 |

Full breakdown in `docs/EXPERIMENT_PROGRESS_AND_FINDINGS.md` and `outputs/eval/render_eval_report.md`.
