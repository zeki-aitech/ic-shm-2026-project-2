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
  - `src/segmentation/` (Task A — 2D pseudo-labeling):
    - `dataset.py`: `BridgeSegDataset` (images + GT masks).
    - `train.py`: fine-tunes SegFormer (mit-b0), validates 2D mIoU on the 60-image holdout.
    - `infer.py`: predicts pseudo-masks for the 100 unlabeled images.
  - `src/gaussian_splatting/` (Task B — **officially scored**, see `docs/EVALUATION_METRICS.md` §0):
    - `undistort.py`: one-time lens-undistortion pass to a pinhole convention.
    - `dataset.py`: `GSCamera` / `build_camera_list`, built from `PycolmapReconstructor` output.
    - `model.py`: `SemanticGaussianModel` — fused RGB + per-class semantic-logit rasterization.
    - `losses.py`: photometric (L1 + D-SSIM) + semantic cross-entropy.
    - `train.py`: training loop using `gsplat.strategy.DefaultStrategy` for densification/pruning.
    - `render.py`: **the contest submission deliverable** — renders RGB + semantic map from any pose.
  - `src/colmap_io/` (camera/pose loading & semantic voting — shared infrastructure for Task B):
    - `models.py`: `CameraIntrinsics`, `ImagePose`, `Point3D` dataclasses.
    - `reconstructor.py`: `PycolmapReconstructor` — LO-RANSAC sparse triangulation.
    - `semantic_voting.py`: `SemanticProjector` — 2D-to-3D multi-view majority voting.
  - `src/evaluation/`:
    - `metrics.py`: confusion matrix / IoU / mIoU, `trajectory_interleaved_split`.
    - `render_metrics.py`: PSNR/SSIM/LPIPS + mIoU on rendered holdout views (official scoring, §0).
  - `tests/`: Automated unit testing suite (`uv run pytest`).
- **Status**:  **Codebase structured, commented, and verified.**

---

### 2. Reproducibility Guide (`README.md`)
- **Requirement**: A comprehensive README explaining exact steps to install dependencies, run the pipeline, and reproduce benchmark results.
- **Repository Alignment**:
  - Root [`README.md`](../README.md) configured with `uv` package management:
    ```bash
    # 1. Install dependencies (gsplat/torch pinned to a working prebuilt-wheel combo, see pyproject.toml)
    uv sync --extra deeplearning

    # 2. Run automated test suite
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest

    # 3. Task A: fine-tune 2D segmenter, pseudo-label the 100 unlabeled images
    uv run python -m src.segmentation.train --epochs 80
    uv run python -m src.segmentation.infer --checkpoint outputs/checkpoints/segformer_mitb0/best.pt

    # 4. Task B: train the Semantic 3D Gaussian Splatting model (officially scored deliverable)
    uv run python -m src.gaussian_splatting.train --iters 20000 --downsample 0.5

    # 5. Render RGB + semantic map from an arbitrary camera pose (contest submission artifact)
    uv run python -m src.gaussian_splatting.render --checkpoint outputs/checkpoints/gaussians/final.pt \
        --pose-line "..." --out-rgb rgb.png --out-sem sem.png

    # 6. Evaluate: PSNR/SSIM/LPIPS + mIoU on the 60 never-trained holdout views
    uv run python -m src.evaluation.render_metrics --checkpoint outputs/checkpoints/gaussians/final.pt
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
  ├── pseudo_masks/                 # 100 AI-predicted masks for unlabeled frames (Task A)
  ├── checkpoints/
  │   ├── segformer_mitb0/best.pt       # Task A segmentation checkpoint
  │   └── gaussians/final.pt            # Task B Semantic Gaussian Splatting checkpoint (scored)
  └── eval/render_eval_report.md    # PSNR/SSIM/LPIPS + mIoU on the 60 holdout views
  ```
- **Action Needed**: Upload trained model checkpoints to Google Drive and generate a public read-only link prior to submission.
- **Status**: 🟡 **Local dataset ready; Cloud link to be generated upon final model export.**

---

### 4. 10-Minute Presentation Video
- **Requirement**: A 10-minute presentation video with **both presentation slides and speaker visible** simultaneously (Picture-in-Picture webcam format).
- **Recommended 10-Minute Script Structure**:
  | Time Window | Section | Key Talking Points |
  | :---: | :--- | :--- |
  | **0:00 – 1:30** | **1. Problem Statement & Challenges** | Official brief: render RGB+semantic from blind test viewpoints; SfM camera noise, slender stay-cable challenges. |
  | **1:30 – 3:30** | **2. Task A: 2D Pseudo-Labeling** | SegFormer (mit-b0) fine-tuning, 240/60 trajectory-interleaved split, pseudo-labeling 100 unlabeled frames. |
  | **3:30 – 6:30** | **3. Task B: Semantic 3D Gaussian Splatting** | Fused RGB + semantic-logit rasterization (`gsplat`), sparse-cloud + voted-class warm start, `DefaultStrategy` densification. |
  | **6:30 – 9:00** | **4. Experimental Results & Visualizations** | Accuracy Score = 0.5×Visual Fidelity (PSNR/SSIM/LPIPS) + 0.5×Semantic mIoU on the 60 holdout views; `render.py` novel-view demo. |
  | **9:00 – 10:00** | **6. Conclusion & SHM Implications** | Digital Twin integration for structural health monitoring (vibration & displacement). |
- **Status**: ⚪ **To be recorded after final experimental results.**

---

### 5. PowerPoint Presentation Slides (`.pptx` / `.pdf`)
- **Requirement**: Professional presentation slide deck matching the 10-minute video presentation.
- **Slide Deck Outline (12–15 Slides)**:
  1. *Title Slide*: Project Title, Team Members, Affiliations, IC-SHM 2026.
  2. *Introduction & Background*: Digital Twins for Cable-Stayed Bridges.
  3. *Problem Decomposition*: Task A (2D pseudo-labeling) + Task B (Semantic 3D Gaussian Splatting).
  4. *Dataset Survey & Camera Geometry*: `SIMPLE_RADIAL` intrinsics, 400 registered frames.
  5. *Task A Methodology*: SegFormer fine-tuning, pseudo-labeling the 100 unlabeled frames.
  6. *Task B Methodology*: Fused RGB+semantic rasterization, sparse-cloud warm start, densification.
  7. *Quantitative Benchmarks*: Accuracy Score (PSNR/SSIM/LPIPS + mIoU) on the 60 holdout views.
  8. *Ablation Studies*: Impact of semantic warm-start, pseudo-mask supervision, densification.
  9. *Interactive Visualizations*: Novel-view RGB + semantic renders vs. ground truth.
  10. *Conclusion & Future Work*: UAV vision vibration & SHM integration.
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
    - Sub-task A: 2D Segmentation & Pseudo-Labeling (SegFormer mit-b0).
    - Sub-task B: Semantic 3D Gaussian Splatting (fused RGB + semantic-logit rasterization),
      initialized from a triangulated sparse cloud and per-point voted semantic class.
  - **Section 4: Experiments & Dataset Setup**:
    - IC-SHM 2026 Dataset breakdown (300 labeled + 100 unlabeled + SfM parameters).
    - Official evaluation metrics: Accuracy Score = 0.5×Visual Fidelity (PSNR/SSIM/LPIPS) +
      0.5×Semantic mIoU, on the 60-view trajectory-interleaved holdout.
  - **Section 5: Results & Discussion**:
    - Quantitative comparison tables (Accuracy Score, per-class IoU).
    - Qualitative novel-view RGB + semantic renderings vs. ground truth.
    - Ablation analysis: semantic warm-start, pseudo-mask supervision, densification.
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
