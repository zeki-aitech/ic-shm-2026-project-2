# IC-SHM 2026 — Required Submission Items & Preparation Guidelines

**Project**: IC-SHM 2026 (Project 2) — Multi-view Semantic 3D Reconstruction of Bridge Structures  
**Source Document**: `docs/required-submission-items.jpeg`  
**Winner Announcement Date**: October 31, 2026  
**Awards**: First Prize ($1,500 USD), Second Prize ($500 USD), Third Prize  
**Language**: English  

---

## 📑 Executive Overview

According to the official competition guidelines extracted from [`docs/required-submission-items.jpeg`](required-submission-items.jpeg), each participating team must deliver a complete submission package comprising:
1. **Commented Python Code**
2. **Reproducibility README**
3. **Dataset Links (Google Drive / Baidu Cloud)**
4. **10-Minute Presentation Video** (with both slides and speaker visible)
5. **PowerPoint Slides**
6. **10–15 Page Academic Paper** (using the official IC-SHM template)

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      OFFICIAL SUBMISSION PACKAGE (6 ITEMS)                  │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ [1] Python Codebase    ──► Clean, well-commented modules in src/ & tests/   │
 │ [2] Reproducibility    ──► Step-by-step reproduction guide in README.md     │
 │ [3] Dataset & Weights  ──► Shareable Google Drive / Baidu Cloud link        │
 │ [4] 10-Min Video       ──► MP4 recording with slides & speaker webcam       │
 │ [5] Presentation Deck  ──► PowerPoint slides (.pptx / .pdf)                 │
 │ [6] Academic Paper     ──► 10–15 page paper using official IC-SHM template  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Item-by-Item Detailed Checklist

### 1. Commented Python Code
- **Requirement**: Full source code implemented in Python with thorough docstrings and inline comments.
- **Repository Alignment**:
  - `src/utils/`: 2D data preprocessing (`json_to_mask.py`, `create_overlay_dataset.py`).
  - `src/reconstruction/`:
    - `models.py`: Strongly-typed dataclasses (`CameraIntrinsics`, `ImagePose`, `Point3D`).
    - `pycolmap_reconstructor.py`: Fast LO-RANSAC 3D triangulation.
    - `gpu_pipeline.py`: GPU-accelerated SIFT extraction and sequential matching on CUDA.
    - `semantic_projector.py`: 2D-to-3D back-projection & multi-view majority voting.
    - `point_cloud_filter.py`: 8-stage structure-aware geometric filtering pipeline (Deck PCA, Tower tube, Cable fan planes).
    - `visualizer.py`: Interactive Plotly and ASCII PLY export.
  - `tests/`: Automated unit testing suite (20 test cases passing).
- **Status**:  **Codebase structured, commented, and verified.**

---

### 2. Reproducibility Guide (`README.md`)
- **Requirement**: A comprehensive README explaining exact steps to install dependencies, run the pipeline, and reproduce benchmark results.
- **Repository Alignment**:
  - Root [`README.md`](../README.md) configured with `uv` package management:
    ```bash
    # 1. Install dependencies
    uv sync --extra deeplearning

    # 2. Run automated test suite
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest

    # 3. Generate 2D ground-truth masks
    uv run python -m src.utils.json_to_mask

    # 4. Run 3D semantic projection
    uv run python -m src.reconstruction.semantic_projector

    # 5. Run structure-aware geometric filtering
    uv run python -m src.reconstruction.point_cloud_filter \
        --input outputs/point_clouds/semantic_bridge_sparse.ply \
        --output outputs/point_clouds/semantic_bridge_filtered.ply
    ```
- **Status**:  **Fully documented in root README.md.**

---

### 3. Dataset Package & Cloud Links
- **Requirement**: Required datasets or a shareable Google Drive / Baidu Cloud link containing raw images, SfM parameters, and generated 3D models.
- **Submission Package Structure**:
  ```text
  submission_dataset_bundle/
  ├── raw_contest_dataset/          # 400 images + camera_parameters/
  ├── gt_masks/                     # 300 ground-truth PNG masks
  ├── pseudo_masks/                 # 100 AI-predicted masks for unlabeled frames
  └── point_clouds/                 # Reconstructed 3D semantic models (.ply)
      ├── semantic_bridge_sparse.ply
      ├── semantic_bridge_gpu.ply
      └── semantic_bridge_filtered.ply
  ```
- **Action Needed**: Upload final `.ply` point clouds and trained model checkpoints to Google Drive and generate a public read-only link prior to submission.
- **Status**: 🟡 **Local dataset ready; Cloud link to be generated upon final model export.**

---

### 4. 10-Minute Presentation Video
- **Requirement**: A 10-minute presentation video with **both presentation slides and speaker visible** simultaneously (Picture-in-Picture webcam format).
- **Recommended 10-Minute Script Structure**:
  | Time Window | Section | Key Talking Points |
  | :---: | :--- | :--- |
  | **0:00 – 1:30** | **1. Problem Statement & Challenges** | Drone photogrammetry, SfM camera noise, slender stay-cable challenges. |
  | **1:30 – 3:30** | **2. Task A: 2D Semantic Segmentation** | Semi-supervised learning (YOLO-seg / SegFormer), handling 100 unlabeled frames. |
  | **3:30 – 5:30** | **3. Task B: 3D Triangulation & Semantic Fusion** | LO-RANSAC DLT triangulation, Look-Up Table (LUT) back-projection, Majority Voting with $>50\%$ cable safeguard. |
  | **5:30 – 7:30** | **4. Structure-Aware 3D Filtering** | Domain priors: Deck 2-pass PCA plane, Tower shaft K-Means clustering, Cable Fan Sheet snapping (Hu et al. 2020). |
  | **7:30 – 9:00** | **5. Experimental Results & Visualizations** | 3D mIoU, Reprojection Error $< 0.8\text{ px}$, Deck Planarity MAD $< 0.05\text{ m}$, 3D interactive Plotly demo. |
  | **9:00 – 10:00** | **6. Conclusion & SHM Implications** | Digital Twin integration for structural health monitoring (vibration & displacement). |
- **Status**: ⚪ **To be recorded after final experimental results.**

---

### 5. PowerPoint Presentation Slides (`.pptx` / `.pdf`)
- **Requirement**: Professional presentation slide deck matching the 10-minute video presentation.
- **Slide Deck Outline (12–15 Slides)**:
  1. *Title Slide*: Project Title, Team Members, Affiliations, IC-SHM 2026.
  2. *Introduction & Background*: Digital Twins for Cable-Stayed Bridges.
  3. *Problem Decomposition*: Two-Task Architecture (Task A + Task B).
  4. *Dataset Survey & Camera Geometry*: `SIMPLE_RADIAL` intrinsics, 400 registered frames.
  5. *Task A Methodology*: Semi-Supervised Segmentation on 100 Unlabeled Frames.
  6. *Task B Triangulation & LUT Projection*: Fast multi-view voting.
  7. *Structure-Aware Geometric Filtering*: Mathematical formulation of bridge priors.
  8. *Quantitative Benchmarks*: 2D/3D mIoU, Reprojection Error, Deck MAD, Cable Deviation.
  9. *Ablation Studies*: Impact of Cable Majority Rule & Geometric Snapping.
  10. *Interactive 3D Visualizations*: Before vs. After filtering point clouds.
  11. *Conclusion & Future Work*: UAV vision vibration & SHM integration.
- **Status**: ⚪ **Template structure planned; slides to be assembled.**

---

### 6. 10–15 Page Academic Paper (IC-SHM Template)
- **Requirement**: Full-length technical research paper (10–15 pages) written in English following the official IC-SHM paper template.
- **Proposed Paper Structure & Section Plan**:
  - **Section 1: Introduction**: Cable-stayed bridge inspection challenges, UAV photogrammetry, motivation.
  - **Section 2: Related Work**:
    - Multi-view Structure-from-Motion (Schönberger et al., Lowe SIFT).
    - 2D/3D Structural Bridge Segmentation (Lin et al. 2025, Hu et al. 2020).
  - **Section 3: Proposed Pipeline Architecture**:
    - Sub-task A: Semi-Supervised 2D Segmentation & Pseudo-Labeling.
    - Sub-task B1: Multi-View LO-RANSAC Triangulation & LUT Semantic Fusion.
    - Sub-task B2: Structure-Aware Geometric Filtering (Deck PCA, Tower Tube, Cable Fan Planes).
  - **Section 4: Experiments & Dataset Setup**:
    - IC-SHM 2026 Dataset breakdown (300 labeled + 100 unlabeled + SfM parameters).
    - Evaluation metrics ($mIoU_{3D}, IoU_{\text{cable}}$, Reprojection Error, Planarity MAD).
  - **Section 5: Results & Discussion**:
    - Quantitative comparison tables.
    - Qualitative 3D point cloud renderings.
    - Ablation analysis on geometric filtering stages.
  - **Section 6: Conclusion**: Summary of findings and practical implications for SHM Digital Twins.
- **Status**: ⚪ **Draft outline established; text to be written using the IC-SHM template.**

---

## 🎯 Deliverable Status Tracker

| Deliverable Item | Target Format | Current Status | Location / Reference |
| :--- | :---: | :---: | :--- |
| **1. Commented Python Code** | `.py` files |  **100% Ready** | [`src/`](../src/) & [`tests/`](../tests/) |
| **2. Reproducibility Guide** | Markdown |  **100% Ready** | [`README.md`](../README.md) |
| **3. Dataset & Cloud Links** | Cloud Link / `.ply` | 🟡 **In Progress** | [`data/`](../data/) $\to$ Google Drive |
| **4. 10-Minute Video** | `.mp4` (1080p) | ⚪ **Planned** | Recording script in Section 4 above |
| **5. Presentation Slides** | `.pptx` / `.pdf` | ⚪ **Planned** | Slide outline in Section 5 above |
| **6. 10–15 Page Paper** | `.pdf` (LaTeX/Word) | ⚪ **Planned** | Outline in Section 6 above |
