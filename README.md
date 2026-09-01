# IC-SHM 2026 — Project 2: Structure-Aware 3D Semantic Point Cloud Reconstruction for Cable-Stayed Bridges

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![pycolmap](https://img.shields.io/badge/pycolmap-CUDA12-orange.svg)](https://github.com/colmap/pycolmap)

## 📌 1. Project Overview
This project addresses the problem of **Structure-Aware 3D Semantic Point Cloud Reconstruction** for **Cable-Stayed Bridges** from multi-view UAV images and Structure-from-Motion (SfM - COLMAP) camera parameters, developed for the **IC-SHM 2026 Competition** (Structural Health Monitoring).

The primary objective is to reconstruct a high-fidelity 3D semantic point cloud (`.ply`) from multi-view drone imagery and 2D polygonal annotations (Labelme JSON / Ground-Truth PNG Masks), applying domain-specific structural and geometric filtering pipelines to remove outliers and accurately preserve slender structural bridge components.

---

## 🏗️ 2. Semantic Class Taxonomy

The semantic segmentation taxonomy consists of 5 structural classes:

| Class ID | Class Name (`label`) | RGB Color | Hex Color | Description & Geometric Characteristics |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `background` | `(128, 128, 128)` | `#808080` | Sky, river, terrain, surroundings |
| **1** | `deck` | `(255, 0, 0)` | `#FF0000` | Bridge deck / roadway girder (planar horizontal surface) |
| **2** | `stay_cable` | `(0, 255, 255)` | `#00FFFF` | Cable stay bundles (arranged along two vertical fan planes) |
| **3** | `tower` | `(0, 255, 0)` | `#00FF00` | Pylons / Tower shafts (vertical column structures) |
| **4** | `foundation` | `(255, 255, 0)` | `#FFFF00` | Bridge piers / substructure footings |

---

## 📂 3. Directory Structure

```text
ic-shm-2026-project-2/
├── .devcontainer/                # Docker container configuration with CUDA 12.1, PyTorch, Pycolmap
│   ├── Dockerfile
│   └── devcontainer.json
├── docs/                         # Technical specifications and contest documentation
│   ├── CONTEST_SPEC_AND_SURVEY.md# Detailed problem statement and dataset survey
│   ├── EVALUATION_METRICS.md     # Mathematical evaluation framework and SHM metrics
│   └── THEORY_COLMAP_AND_SIFT.md # Complete handbook on SIFT, Epipolar Geometry & Triangulation
├── notebooks/                    # Interactive 3D visualization notebooks
│   ├── 00_understand_colmap_outputs.ipynb # Complete tutorial & visualizer of raw COLMAP outputs
│   ├── 01_visualize_semantic_3d.ipynb     # Interactive semantic 3D point cloud visualizer
│   └── 02_visualize_colmap_mapping_2d_3d.ipynb # 2D-3D mapping & voting visualizer
├── src/
│   ├── reconstruction/           # 3D Reconstruction & Semantic Pipeline Core
│   │   ├── models.py                  # Core dataclasses (CameraIntrinsics, ImagePose, Point3D)
│   │   ├── pycolmap_reconstructor.py  # LO-RANSAC triangulation via pycolmap (GPU-accelerated)
│   │   ├── gpu_pipeline.py            # GPU SIFT feature extraction + sequential matching pipeline
│   │   ├── semantic_projector.py      # 2D-to-3D back-projection with Multi-view Majority Voting
│   │   ├── point_cloud_filter.py      # Multi-stage structure-aware geometric point cloud filter
│   │   └── visualizer.py              # PLY reader & interactive Plotly 3D visualizer
│   └── utils/                    # 2D Data Preprocessing
│       ├── json_to_mask.py            # Convert Labelme JSON annotations to 8-bit PNG masks
│       └── create_overlay_dataset.py  # Generate visual overlay dataset with class legends
└── tests/                        # Automated unit tests
    ├── test_models.py
    ├── test_point_cloud_filter.py
    ├── test_semantic_projector.py
    └── test_visualizer.py
```

---

## ⚙️ 4. Technical Pipeline Architecture

1. **2D Mask Preprocessing (`json_to_mask.py`)**:
   - Converts Labelme polygon annotations into 8-bit indexed masks.
   - Drawing order priority: $\text{deck (1)} \to \text{tower (3)} \to \text{foundation (4)} \to \mathbf{\text{stay\_cable (2)}}$ to prevent thin cables from being occluded by surrounding regions.

2. **3D Triangulation & GPU Acceleration (`pycolmap_reconstructor.py` / `gpu_pipeline.py`)**:
   - Uses `pycolmap-cuda12` for GPU SIFT extraction and sequential matching with fixed camera intrinsics, triangulating a dense sparse point cloud ($>80,000$ points).

3. **Multi-View 2D-to-3D Semantic Back-Projection (`semantic_projector.py`)**:
   - **Multi-view Majority Voting**: Votes on class label per 3D point across all observing camera frames.
   - Slender structures handling: `stay_cable` requires strict absolute majority ($>50\%$), and non-cable ties are resolved via `TIE_BREAK_PRIORITY` (`tower` > `foundation` > `deck` > `background`).

4. **Structure-Aware Geometric Filtering Pipeline (`point_cloud_filter.py`)**:
   - **Stage 1 (Background Drop)**: Removes class 0 (`background`).
   - **Stage 2 (Class-Specific SOR)**: Applies Statistical Outlier Removal independently per structural component with tuned thresholds.
   - **Stage 3 (Deck Plane Residuals)**: Fits a 2-pass PCA plane to deck points with MAD residual rejection.
   - **Stage 4 (Deck Core Density)**: k-NN density filtering in the $(u, w)$ plane to remove sparse coplanar roadway outliers.
   - **Stage 5 (Tower Core Tube)**: Clusters tower shafts along the longitudinal axis and keeps points within a narrow $(u, w)$ tube per shaft.
   - **Stage 6 (Stay-Cable Structural Envelope)**: Enforces height bounds (above deck, below tower apex) and longitudinal spans.
   - **Stage 7 (Stay-Cable Left/Right Fan Planes)**: Filters cables based on distance to the two tower-anchored vertical fan sheets.
   - **Stage 8 (Optional Cable Snapping)**: `project_cables_to_fan_planes` snaps cable points perpendicularly onto the nearest fan plane for CAD/BIM alignment.

---

## 🚀 5. Quickstart Guide

### Environment Setup
We use `uv` for modern, fast Python package management.
```bash
# Install dependencies and sync virtual environment
uv sync

# (Optional) Install deeplearning group for segmentation
uv sync --extra deeplearning
```

### Running Unit Tests
```bash
uv run pytest
```

### Running the Point Cloud Filter Pipeline
```bash
python3 -m src.reconstruction.point_cloud_filter \
    --input outputs/point_clouds/semantic_bridge_gpu.ply \
    --output outputs/point_clouds/semantic_bridge_filtered.ply \
    --colmap-model outputs/gpu_pipeline/triangulated
```

### Interactive 3D Visualization
Launch JupyterLab and run [`notebooks/01_visualize_semantic_3d.ipynb`](notebooks/01_visualize_semantic_3d.ipynb) to inspect and interact with the 3D colored point cloud.
