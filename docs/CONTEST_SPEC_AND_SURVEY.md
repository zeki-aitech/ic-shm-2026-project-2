# IC-SHM 2026 (Project 2) — Technical Specifications & Problem Survey

**Project**: Multi-view Semantic 3D Reconstruction of Bridge Structures
**Survey Date**: September 2026

> ## ⚠️ Official Source of Truth
> The authoritative task definition and scoring rule for Project 2 are stated verbatim in
> `data/Contest Dataset/The 4th International Project Competition for SHM_2026.pdf`, pp. 9-10
> ("Project 2: Multi-view Semantic 3D Reconstruction of Bridge Structures"):
>
> - **Task**: *"Participants must develop a model that reconstructs a semantically labeled 3D
>   representation of a bridge structure from multi-view images."*
> - **Goal & Evaluation**: submitted models are evaluated on a **separate blind test set** of
>   camera viewpoints on two criteria:
>   1. **Visual fidelity** — the reconstructed 3D model is rendered from the test viewpoints and
>      compared with the original photos via **PSNR, SSIM, and LPIPS**.
>   2. **Semantic accuracy** — the reconstructed 3D model is rendered into semantic maps (official
>      class IDs) from the test viewpoints and compared with GT via **mIoU**.
>   $$\text{Accuracy Score} = 0.50 \times \text{Visual Fidelity Score} + 0.50 \times \text{Semantic mIoU Score}$$
> - **Submission requirement**: any reconstruction methodology is allowed (point-based, mesh-based,
>   implicit neural representations, or **3D Gaussian Splatting**), but the submitted model/script
>   must be able to generate **both** an RGB image and a semantic map from the provided test
>   viewpoints, using official class IDs, so the organizers can evaluate automatically.
>
> **Everything below this notice that references a 3D point-cloud output, $mIoU_{3D}$ measured
> directly on point classifications, or the Pillar-1/2/3 SHM-prior metric targets was this
> repository's own early-stage survey/interpretation, written before the official PDF brief was
> available in this repo — it does **not** define the official scoring rule above, and the
> point-cloud reconstruction + geometric-filtering pipeline it describes (deck plane fitting,
> tower core tube, cable fan planes) has since been **removed from this repository** rather than
> kept as a parallel deliverable. What remains from that earlier work is the camera/pose loading
> and triangulation infrastructure (`src/colmap_io/`), which the current pipeline reuses to load
> posed training views and to warm-start the Gaussian Splatting model's parameters from a real
> triangulated, semantically-voted sparse cloud instead of random initialization (see
> `src/gaussian_splatting/model.py::init_from_sparse`). The actual implementation matching the
> official brief lives in `src/gaussian_splatting/` (3D reconstruction + rendering) and
> `src/segmentation/` (2D pseudo-labeling of the unlabeled frames) — see the root `README.md`
> for the pipeline and CLI commands.**

---

## I. PROBLEM STATEMENT & OBJECTIVES (historical survey — superseded by the official PDF above)

### 1. Primary Objective
The core objective of **IC-SHM 2026 Project 2** is to build a high-fidelity **As-is 3D Semantic Digital Twin** of a real-world cable-stayed bridge by fusing multi-view drone photogrammetry, Structure-from-Motion (SfM) camera poses, and 2D semantic image segmentations.

The output is a clean, labeled 3D point cloud (`.ply`) where every spatial coordinate $(x, y, z)$ is assigned to its exact structural bridge component (`deck`, `stay_cable`, `tower`, `foundation`, or `background`) with rigorous geometric fidelity.

---

### 2. Two-Task Problem Decomposition (this repo's implementation — reframed to match the official rendering-based Accuracy Score)

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      TASK A: 2D SEMANTIC SEGMENTATION                       │
 │  • src/segmentation/: fine-tunes SegFormer (mit-b0) on the 240-image        │
 │    trajectory-interleaved train split                                       │
 │  • Input: Labeled UAV images (300 frames) with Labelme polygon JSON masks   │
 │  • Output: Pseudo-masks for the 100 unlabeled frames (outputs/pseudo_masks) │
 │  • Metric: 2D mIoU on the 60-image holdout (paper-reported, not organizer-  │
 │    scored - Project 2 has no standalone 2D-mIoU criterion in the PDF)       │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼ Widens semantic supervision for Task B
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │       TASK B: SEMANTIC 3D GAUSSIAN SPLATTING (officially scored task)       │
 │  • src/gaussian_splatting/: fused RGB + per-class semantic-logit            │
 │    rasterization (gsplat), trained on 240 GT-mask + 100 pseudo-mask views   │
 │  • Gaussian means/colors/semantic-logits warm-started from this repo's      │
 │    triangulated + semantically-voted sparse cloud (see Section III below)   │
 │  • Output: render.py renders RGB + semantic map from ANY camera viewpoint   │
 │  • Metric: Accuracy Score = 0.5*Visual Fidelity (PSNR/SSIM/LPIPS) +         │
 │    0.5*Semantic mIoU, evaluated on the 60 never-trained holdout views       │
 └─────────────────────────────────────────────────────────────────────────────┘
```

| Task | Core Objective | Inputs | Deliverables & Metrics |
| :---: | :--- | :--- | :--- |
| **Task A: 2D Pseudo-Labeling** | Fine-tune a 2D segmenter to pseudo-label the 100 unlabeled frames, widening Task B's semantic supervision. | `images/` + `json/` (Labelme polygons), `unlabeled_Images/` | `outputs/pseudo_masks/*.png`; 2D mIoU on 60-image holdout (internal/paper metric). |
| **Task B: Semantic 3D Gaussian Splatting** | Reconstruct a model that renders RGB + semantic maps from arbitrary viewpoints. | `camera_parameters/`, GT + pseudo masks | `render.py` CLI; **officially scored**: Accuracy Score on 60 holdout views. |

---

### 3. Core Technical Challenges
1. **Geometric Disparity Among Structural Components**:
   - **Bridge Deck (`deck`)**: Large planar surface spanning horizontally.
   - **Towers / Pylons (`tower`)**: Tall vertical columns.
   - **Foundations (`foundation`)**: Substructures located at the lower boundary.
   - **Stay Cables (`stay_cable`)**: Slender, thin linear features occupying few pixels. During 3D triangulation and back-projection, feature points on cables often suffer from background bleeding (sky/water pixels misclassified as cables, or thin cables occluded by deck/tower).
2. **Triangulation Noise & Drift**:
   - The initial SfM dataset provides 2D feature tracks without precomputed 3D coordinates (`points3D.txt`). Narrow baseline triangulation from UAV flight lines causes depth drift and floating outliers.
3. **Semi-Supervised Label Propagation**:
   - Only a subset of UAV images contain manual 2D polygon annotations (Labelme JSON), while the remaining images are unlabeled. Multi-view voting and geometric constraints are necessary to clean and propagate labels into 3D.

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

The provided SfM dataset (generated via COLMAP) provides 4 purely geometric core outputs. Note that COLMAP is semantically blind (it outputs a geometric world with $x, y, z$ coordinates but no component labels):

1. **Camera Intrinsics (`cameras.txt`)**:
   - Model: `SIMPLE_RADIAL` | Resolution: $1320 \times 989$ px.
   - Focal Length: $f \approx 925.7$ px | Principal Point: $c_x = 660.0, c_y = 494.5$ px | Radial Distortion: $k_1$.
   - *Purpose*: Enables precise mathematical 3D-to-2D projection.
2. **Camera Extrinsics (`images.txt`)**:
   - The 6-DOF poses (Rotation $R$, Translation $T$) of the UAV for all 400 frames in world space.
3. **Sparse 3D Point Cloud**:
   - Spatial coordinates $(x,y,z)$ and RGB colors for bridge points. This raw output is highly noisy around slender structures like stay cables.
4. **2D-3D Feature Tracks (86,336 tracks)**:
   - The critical mapping table that links a specific 3D point $X_i$ to its observed 2D pixel coordinates $(u, v)$ across multiple UAV frames.

> **The Geometric-to-Semantic Bridge**: The 2D-3D feature tracks (Output #4) act as the link between Task A and Task B. The pipeline looks up a 3D point's observed 2D pixels, queries the AI-predicted 2D semantic masks (from Task A) at those pixels, and "copies" the label back to the 3D point (Semantic Fusion).

### 3. Role of the COLMAP / pycolmap Pipeline in This Codebase

The `pycolmap` module in our codebase serves three distinct functional roles depending on the pipeline configuration:

1. **Mandatory 3D Point Triangulation (LO-RANSAC)**:
   - **Why It Is Required**: The contest dataset delivers camera intrinsics (`cameras.txt`) and 2D feature observations (`images.txt`), but **does not contain precomputed 3D coordinates (`points3D.txt`)**.
   - **Action**: `src/colmap_io/reconstructor.py` (`PycolmapReconstructor`) solves multi-view optical ray intersections using **LO-RANSAC Triangulation** to compute $(X, Y, Z)$ coordinates and reprojection errors for all 86,336 feature tracks.
2. **Camera Pose Refinement & Calibration (Bundle Adjustment)**:
   - **Why It Is Useful**: As stated in the contest `README.md`, the provided camera parameters are reference SfM estimates rather than ground truth.
   - **Action**: The pipeline can run **Global / Local Bundle Adjustment (BA)** to jointly refine focal length $f$, distortion $k_1$, and 6-DOF camera poses ($R, T$) to minimize residual reprojection drift below $1.0\text{ px}$.
3. **Dense Surface Reconstruction (MVS / PatchMatch Stereo)**:
   - **Why It Is Useful**: To scale from a sparse cloud (86,336 points) to a dense photorealistic Digital Twin with millions of surface points across the roadway and cables.
   - **Action**: Invokes COLMAP's `patch_match_stereo` and `stereo_fusion` modules.

| Operational Mode | Pipeline Tasks Executed | Typical Runtime | Target Use Case |
| :--- | :--- | :---: | :--- |
| **Mode 1: Fast Triangulation (Default)** | Reuse contest poses $\to$ Triangulate $(X, Y, Z)$ via LO-RANSAC | **~3–5 seconds** | Rapid prototyping, fast verification of semantic fusion |
| **Mode 2: Pose Refinement** | Bundle Adjustment on contest poses $\to$ Triangulate | **~1–2 minutes** | Improved geometric accuracy ($< 1.0\text{ px}$ error) |
| **Mode 3: Full SfM from Scratch** | SIFT extraction $\to$ Sequential/Exhaustive matching $\to$ Global SfM | **~10–15 minutes** | Complete custom calibration & experimental re-estimation |

### 4. Semantic Taxonomy & Color Codes

| Class ID | Component Name | RGB Color | Hex Color | Geometric Role |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `background` | `(128, 128, 128)` | `#808080` | Sky, clouds, water, terrain, non-structural context |
| **1** | `deck` | `(255, 0, 0)` | `#FF0000` | Roadway girder / main bridge deck |
| **2** | `stay_cable` | `(0, 255, 255)` | `#00FFFF` | Cable stay bundles connecting deck to towers |
| **3** | `tower` | `(0, 255, 0)` | `#00FF00` | Main towers / vertical pylons |
| **4** | `foundation` | `(255, 255, 0)` | `#FFFF00` | Piers, abutments, footings |

---

## III. ARCHITECTURE & ADVANCED ALGORITHMS (historical survey; §1-2 still implemented, §3 removed)

### 1. 2D Mask Generation & Drawing Order (`src/utils/json_to_mask.py`) — still implemented
Polygon masks from Labelme JSON are rendered in strict ascending priority:
$$\text{deck (1)} \longrightarrow \text{tower (3)} \longrightarrow \text{foundation (4)} \longrightarrow \mathbf{\text{stay\_cable (2)}}$$
Drawing `stay_cable` on the top layer ensures thin cable lines are never overwritten by broader deck or pylon masks.

### 2. 2D-to-3D Back-Projection & Multi-View Voting (`src/colmap_io/semantic_voting.py`) — still implemented
- For every triangulated 3D point $X_i$, retrieve all 2D observations $(u_{ik}, v_{ik})$ across observing cameras $I_k$.
- Sample 2D class label $L_{ik} = \text{Mask}_k(u_{ik}, v_{ik})$.
- **Voting Decision Rules**:
  - `stay_cable` requires **strict absolute majority** ($> 50\%$).
  - If cable count fails to reach $>50\%$, cable votes are dropped and plurality voting applies to remaining classes.
  - Ties are broken via `TIE_BREAK_PRIORITY` (`tower` > `foundation` > `deck` > `background`).
- Used today to warm-start the Gaussian Splatting model's per-Gaussian semantic logits (see `src/gaussian_splatting/model.py::init_from_sparse`), not to export a standalone point cloud.

### 3. Structure-Aware 3D Geometric Filtering Pipeline — **removed from this repository**
This subsection previously described `point_cloud_filter.py` (deck plane fitting, tower core
tube, cable fan-plane snapping) in detail. That module, its 2D-mask-refinement notebooks, and
the PLY-export/visualization tooling around it have been deleted as part of the pivot to the
render-based Gaussian Splatting pipeline (see the banner at the top of this document). The
algorithm descriptions below are kept only as a reference for anyone who wants to reintroduce a
similar geometric-filtering analysis in the future.

1. **Bridge-Local Coordinate Frame**:
   - Estimates world gravity / vertical axis $v$ from UAV camera roll statistics (near-zero roll).
   - Computes longitudinal axis $u$ along the deck span and lateral axis $w$ perpendicular to span ($w = u \times v$).
2. **Deck Plane Fitting**:
   - 2-pass PCA plane fitting on deck points with Median Absolute Deviation (MAD) residual thresholding.
3. **Deck Core Density**:
   - k-NN density estimation in the bridge $(u, w)$ plane to prune coplanar outliers beyond the roadway boundary.
4. **Tower Shaft Core Tube**:
   - Clusters tower shafts along the longitudinal axis via 1D K-Means, then bounds points within a narrow $(u, w)$ tube per shaft.
5. **Stay-Cable Structural Envelope**:
   - Enforces physical bounding volumes: height restricted to $[h_{\text{deck}}, h_{\text{tower\_top}}]$ and span restricted to bridge corridor.
6. **Tower-Anchored Fan Planes**:
   - Identifies lateral offsets $d_{\text{left}}, d_{\text{right}}$ of the two cable fan sheets from tower face percentiles. Removes cables exceeding lateral deviation tolerance $\tau$.
7. **Geometric Cable Snapping**:
   - Projects cable points perpendicularly onto the nearest fan plane along axis $w$, preserving true elevation $z$ and producing planar sheets for CAD/BIM modeling.

---

## IV. RESEARCH REFERENCES (Lin 2025, Hu 2020)

The problem decomposition and geometric filtering logic are strongly grounded in prior research:
1. **Lin et al. (2025)**: *A structure-oriented loss function for automated semantic segmentation of bridges*. This paper emphasizes structural boundaries and semantic loss during 2D segmentation, forming the basis for our Task A methodology and mIoU evaluation.
2. **Hu et al. (2020)**: *Structure-aware 3D reconstruction for cable-stayed bridges: A learning-based method*. This paper details structure-aware reconstruction techniques, geometric constraints, and multi-view consistency, forming the core inspiration for Task B.

---

## V. FUTURE ROADMAP & RESEARCH EXTENSIONS

1. **Automated Deep Learning 2D Segmentation**: Incorporate zero-shot models (SAM / HQ-SAM) or fine-tuned segmentation models (SegFormer, Mask2Former) to automatically generate high-quality masks for the unlabeled UAV frames.
2. **Centralized Configuration**: Decouple file paths into a structured `configs/config.yaml` with unified environment management.
3. **Dynamic Vibration & Displacement Integration**: Fuse reconstructed 3D semantic models with UAV video vision displacement measurements (Project 1) for full-lifecycle structural health monitoring.
