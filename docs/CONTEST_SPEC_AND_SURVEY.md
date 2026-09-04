# IC-SHM 2026 (Project 2) — Technical Specifications & Problem Survey

**Project**: Multi-view Semantic 3D Reconstruction of Bridge Structures

## I. PROBLEM STATEMENT & OBJECTIVES

### 1. Primary Objective

Per `data/Contest Dataset/The 4th International Project Competition for SHM_2026.pdf`
(pp. 9-10), the task is: *"Participants must develop a model that reconstructs a semantically
labeled 3D representation of a bridge structure from multi-view images."*

Submitted models are evaluated on a **separate blind test set** of camera viewpoints on two
criteria:
1. **Visual fidelity** — the reconstructed 3D model is rendered from the test viewpoints and
   compared with the original photos via **PSNR, SSIM, and LPIPS**.
2. **Semantic accuracy** — the reconstructed 3D model is rendered into semantic maps (official
   class IDs) from the test viewpoints and compared with GT via **mIoU**.

$$\text{Accuracy Score} = 0.50 \times \text{Visual Fidelity Score} + 0.50 \times \text{Semantic mIoU Score}$$

The submitted model/script must be able to generate **both** an RGB image and a semantic map
from any given test viewpoint, using official class IDs, so the organizers can evaluate
automatically. See the root `README.md` for the full pipeline and CLI commands.

---

### 2. Two-Task Problem Decomposition

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      TASK A: 2D SEMANTIC SEGMENTATION                       │
 │  • src/segmentation/: fine-tunes SegFormer (mit-b0) on the 240-image        │
 │    trajectory-interleaved train split                                       │
 │  • Input: Labeled UAV images (300 frames) with Labelme polygon JSON masks   │
 │  • Output: Pseudo-masks for the 100 unlabeled frames (outputs/pseudo_masks) │
 │  • Metric: 2D mIoU on the 60-image holdout                                  │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼ Widens semantic supervision for Task B
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                TASK B: SEMANTIC 3D GAUSSIAN SPLATTING                       │
 │  • src/gaussian_splatting/: fused RGB + per-class semantic-logit            │
 │    rasterization (gsplat), trained on 240 GT-mask + 100 pseudo-mask views   │
 │  • Gaussian means/colors/semantic-logits warm-started from a triangulated,  │
 │    semantically-voted sparse cloud (see Section III below)                  │
 │  • Output: render.py renders RGB + semantic map from ANY camera viewpoint   │
 │  • Metric: Accuracy Score = 0.5*Visual Fidelity (PSNR/SSIM/LPIPS) +         │
 │    0.5*Semantic mIoU, evaluated on the 60 never-trained holdout views       │
 └─────────────────────────────────────────────────────────────────────────────┘
```

| Task | Core Objective | Inputs | Deliverables & Metrics |
| :---: | :--- | :--- | :--- |
| **Task A: 2D Pseudo-Labeling** | Fine-tune a 2D segmenter to pseudo-label the 100 unlabeled frames, widening Task B's semantic supervision. | `images/` + `json/` (Labelme polygons), `unlabeled_Images/` | `outputs/pseudo_masks/*.png`; 2D mIoU on the 60-image holdout. |
| **Task B: Semantic 3D Gaussian Splatting** | Reconstruct a model that renders RGB + semantic maps from arbitrary viewpoints. | `camera_parameters/`, GT + pseudo masks | `render.py` CLI; Accuracy Score on the 60-image holdout. |

---

### 3. Core Technical Challenges
1. **Geometric Disparity Among Structural Components**:
   - **Bridge Deck (`deck`)**: Large planar surface spanning horizontally.
   - **Towers / Pylons (`tower`)**: Tall vertical columns.
   - **Foundations (`foundation`)**: Substructures located at the lower boundary.
   - **Stay Cables (`stay_cable`)**: Slender, thin linear features occupying few pixels. Background bleeding (sky/water pixels misclassified as cables) and occlusion by the deck/tower are the main sources of error.
2. **Triangulation Noise & Drift**:
   - The SfM dataset provides 2D feature tracks without precomputed 3D coordinates (`points3D.txt`). Narrow-baseline triangulation from UAV flight lines can cause depth drift and floating outliers.
3. **Semi-Supervised Label Propagation**:
   - Only a subset of UAV images carry manual 2D polygon annotations (Labelme JSON); the rest are unlabeled. Multi-view voting is used to propagate labels for warm-starting the 3D model's semantics.

---

## II. CONTEST DATASET SPECIFICATION

### 1. Dataset Directory Layout
```text
Contest Dataset/
├── camera_parameters/          # SfM parameters & camera poses
│   ├── cameras.txt             # Camera intrinsic parameters
│   ├── images.txt              # Extrinsic poses & 2D feature track observations
│   ├── rigs.txt                # Camera rig configurations
│   └── frames.txt              # Frame timestamps
├── images/                     # ~300+ labeled UAV images
├── unlabeled_Images/           # ~100 unlabeled UAV images
└── json/                       # Labelme JSON polygon annotations for 'images'
```

### 2. Camera & SfM Parameters (COLMAP Outputs)

The provided SfM dataset (generated via COLMAP) provides purely geometric outputs — COLMAP is
semantically blind, producing $x, y, z$ coordinates but no component labels:

1. **Camera Intrinsics (`cameras.txt`)**:
   - Model: `SIMPLE_RADIAL` | Resolution: $1320 \times 989$ px.
   - Focal Length: $f \approx 925.7$ px | Principal Point: $c_x = 660.0, c_y = 494.5$ px | Radial Distortion: $k_1$.
2. **Camera Extrinsics (`images.txt`)**:
   - The 6-DOF poses (Rotation $R$, Translation $T$) of the UAV for all 400 frames in world space.
3. **2D-3D Feature Tracks (86,336 tracks)**:
   - Links each 3D point $X_i$ to its observed 2D pixel coordinates $(u, v)$ across multiple UAV frames. No precomputed 3D coordinates are shipped, so these tracks are triangulated directly.

### 3. Role of the COLMAP / pycolmap Pipeline in This Codebase

- **Mandatory 3D Point Triangulation (LO-RANSAC)**: since the dataset ships camera intrinsics
  (`cameras.txt`) and 2D feature observations (`images.txt`) but no precomputed 3D coordinates,
  `src/colmap_io/reconstructor.py` (`PycolmapReconstructor`) solves multi-view optical ray
  intersections using LO-RANSAC to compute $(X, Y, Z)$ coordinates and reprojection errors for
  all 86,336 feature tracks. This runs in a few seconds and gives the sparse cloud used to
  warm-start the Gaussian Splatting model's parameters.

### 4. Semantic Taxonomy & Color Codes

| Class ID | Component Name | RGB Color | Hex Color | Geometric Role |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `background` | `(128, 128, 128)` | `#808080` | Sky, clouds, water, terrain, non-structural context |
| **1** | `deck` | `(255, 0, 0)` | `#FF0000` | Roadway girder / main bridge deck |
| **2** | `stay_cable` | `(0, 255, 255)` | `#00FFFF` | Cable stay bundles connecting deck to towers |
| **3** | `tower` | `(0, 255, 0)` | `#00FF00` | Main towers / vertical pylons |
| **4** | `foundation` | `(255, 255, 0)` | `#FFFF00` | Piers, abutments, footings |

---

## III. ARCHITECTURE

### 1. 2D Mask Generation & Drawing Order (`src/utils/json_to_mask.py`)
Polygon masks from Labelme JSON are rendered in strict ascending priority:
$$\text{deck (1)} \longrightarrow \text{tower (3)} \longrightarrow \text{foundation (4)} \longrightarrow \mathbf{\text{stay\_cable (2)}}$$
Drawing `stay_cable` on the top layer ensures thin cable lines are never overwritten by broader deck or pylon masks.

### 2. 2D-to-3D Back-Projection & Multi-View Voting (`src/colmap_io/semantic_voting.py`)
- For every triangulated 3D point $X_i$, retrieve all 2D observations $(u_{ik}, v_{ik})$ across observing cameras $I_k$.
- Sample 2D class label $L_{ik} = \text{Mask}_k(u_{ik}, v_{ik})$.
- **Voting Decision Rules**:
  - `stay_cable` requires **strict absolute majority** ($> 50\%$).
  - If cable count fails to reach $>50\%$, cable votes are dropped and plurality voting applies to remaining classes.
  - Ties are broken via `TIE_BREAK_PRIORITY` (`tower` > `foundation` > `deck` > `background`).
- These per-point votes warm-start the Gaussian Splatting model's per-Gaussian semantic logits (see `src/gaussian_splatting/model.py::init_from_sparse`).

### 3. Semantic 3D Gaussian Splatting (`src/gaussian_splatting/`)
- Each Gaussian carries a mean position, scale, rotation quaternion, RGB color, and a 5-class
  semantic logit vector.
- RGB and semantic channels are rendered in a single fused `gsplat.rasterization()` call by
  concatenating them into one `colors` tensor.
- Training minimizes photometric loss (L1 + D-SSIM) plus semantic cross-entropy, using
  `gsplat.strategy.DefaultStrategy` for gradient-driven Gaussian densification/pruning.
- `render.py` renders an RGB image + semantic map from any camera pose.

---

## IV. RESEARCH REFERENCES (Lin 2025, Hu 2020)

1. **Lin et al. (2025)**: *A structure-oriented loss function for automated semantic segmentation of bridges*. Emphasizes structural boundaries and semantic loss during 2D segmentation — relevant to Task A's methodology and mIoU evaluation.
2. **Hu et al. (2020)**: *Structure-aware 3D reconstruction for cable-stayed bridges: A learning-based method*. Structure-aware reconstruction techniques and multi-view consistency — relevant to Task B.

---

## V. FUTURE ROADMAP & RESEARCH EXTENSIONS

1. **Camera Pose Refinement**: joint bundle adjustment / per-image pose optimization during Gaussian Splatting training to improve reconstruction sharpness beyond the provided reference SfM poses.
2. **Full-Resolution Training**: train at native 1320x989 resolution (currently downsampled for iteration speed) for higher Visual Fidelity scores.
3. **Dynamic Vibration & Displacement Integration**: fuse the reconstructed 3D semantic model with UAV video vision displacement measurements (Project 1) for full-lifecycle structural health monitoring.
