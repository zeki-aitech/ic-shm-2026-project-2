# IC-SHM 2026 (Project 2) — Technical Progress, Benchmarks & Scientific Findings

**Project**: Multi-view Semantic 3D Reconstruction of Bridge Structures
**Language**: English (Repository Standard)

---

## 📑 1. Executive Summary & Architecture Overview

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         COMPLETE SYSTEM ARCHITECTURE                        │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ [Step 1: LO-RANSAC Sparse Triangulation]                                    │
 │   ➔ Triangulates the 400 posed UAV frames into an 84,613-point sparse cloud │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ [Step 2: Task A — 2D Pseudo-Labeling]                                       │
 │   ➔ SegFormer (mit-b0) fine-tuned on 240 labeled train images               │
 │   ➔ Pseudo-labels the 100 unlabeled images                                  │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ [Step 3: Semantic Warm-Start]                                               │
 │   ➔ Multi-view majority voting assigns each sparse 3D point a class         │
 │   ➔ Warm-starts Gaussian means/colors/semantic logits (no random init)      │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ [Step 4: Task B — Semantic 3D Gaussian Splatting]                           │
 │   ➔ Fused RGB + semantic-logit rasterization, trained on 340 views          │
 │   ➔ gsplat.strategy.DefaultStrategy for densification/pruning               │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ [Step 5: Trajectory-Interleaved 80/20 Hold-Out Evaluation]                  │
 │   ➔ Render-based PSNR/SSIM/LPIPS + mIoU on 60 never-trained viewpoints      │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 2. End-to-End Run (RTX 3080, 10GB)

| Stage | Configuration | Result |
| :--- | :--- | :--- |
| Task A (SegFormer mit-b0) | 240 train / 60 holdout, 80 epochs, batch 8 | **Val 2D mIoU = 81.27%** (`outputs/checkpoints/segformer_mitb0/best.pt`) |
| Task A inference | 100 unlabeled images | `outputs/pseudo_masks/301..400.png` |
| Task B (Semantic 3DGS) | 340 train views (240 GT + 100 pseudo), 40,000 iters, full resolution (1320x989), `gsplat.strategy.DefaultStrategy`, 84,613 → 602,363 Gaussians | ~47 min wall-clock training |
| Task B evaluation | `render_metrics` on 60 never-trained holdout views | See table below |

**Render-based evaluation (`outputs/eval/render_eval_report.md`):**

| Metric | Value |
| :--- | :---: |
| PSNR | 22.18 dB |
| SSIM | 0.849 |
| LPIPS | 0.334 |
| **Semantic mIoU (structural, 4 classes)** | **91.47%** |
| — deck | 95.09% |
| — stay_cable | 92.13% |
| — tower | 91.13% |
| — foundation | 87.52% |
| Illustrative Visual Fidelity | 0.716 |
| Illustrative Accuracy Score | 0.816 |

Camera poses are used as provided by the contest's COLMAP reconstruction (frozen, no bundle
adjustment) - the dataset README notes they are "reference only," and this remains a direction
with headroom for further improvement.

---

## 🔬 3. Key Scientific Insights

### 3.1 Semantic Warm-Start Accelerates Convergence
Initializing each Gaussian's semantic logits from its multi-view-voted class (a scaled one-hot
at the voted class, rather than random logits) means the semantic channel starts from a
reasonable prior instead of noise, letting the cross-entropy loss focus on refining boundaries
rather than learning the class distribution from scratch.

### 3.2 Fused RGB+Semantic Rasterization
Concatenating RGB(3) and semantic logits(5) into one 8-channel `colors` tensor and rasterizing
once (rather than running the rasterizer twice) reuses the same depth-sorted alpha composite for
both outputs, which is both simpler and faster than a two-pass approach.

### 3.3 Trajectory-Interleaved Holdout Avoids Spatial Leakage
Adjacent UAV frames share >99% visual overlap. A random train/holdout split risks placing
near-duplicate frames on both sides, inflating the measured score. Striding every 5th frame
(`trajectory_interleaved_split`) spreads the holdout evenly across the flight path instead.

---

## 🎯 4. Deliverables Checklist

| Deliverable Item | Target Requirement | Current Status | Next Action |
| :--- | :--- | :---: | :--- |
| **[1] Python Codebase** | Clean, well-commented modules in `src/` & `tests/` |  **Complete** | Unit tests passing (`uv run pytest`); all modules verified. |
| **[2] Reproducibility** | Step-by-step reproduction guide in `README.md` |  **Complete** | Configured with `uv` package manager. |
| **[3] Dataset Package** | Google Drive / Baidu Cloud bundle | 🟡 **In Progress** | Trained checkpoints ready; cloud links to be generated. |
| **[4] 10-Min Video** | MP4 video with slides & speaker PiP webcam | ⏳ **Upcoming** | Recording after final paper draft. |
| **[5] Presentation Deck** | PowerPoint slides (`.pptx` / `.pdf`) | ⏳ **Upcoming** | Slide outline structured based on the Accuracy Score results. |
| **[6] Academic Paper** | 10–15 page paper using official IC-SHM template | ⏳ **Upcoming** | Ready to draft sections using data and tables from this document. |
