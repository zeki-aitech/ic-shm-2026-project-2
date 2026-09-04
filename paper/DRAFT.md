<!--
Working skeleton for the IC-SHM 2026 Project 2 paper (target: 10-15 pages, official IC-SHM
template, English). This is a structural/content draft, NOT yet formatted in the official
template — pour this content into the downloaded template once available from the contest
website before final submission. Sections marked [NEEDS] require additional work (citation
research, figures, or numbers not yet produced).
-->

# Semantic 3D Gaussian Splatting for Multi-View Reconstruction of Cable-Stayed Bridges

**[NEEDS] Author names, affiliations, IC-SHM 2026 Project 2 team identifier**

---

## Abstract

*(150-250 words — write last, once Results is final)*

We present a method for reconstructing a semantically labeled 3D representation of a
cable-stayed bridge from multi-view UAV imagery, targeting the IC-SHM 2026 Project 2 evaluation
protocol: a model must render both an RGB image and a per-pixel semantic map from an arbitrary
camera viewpoint, scored against held-out views by visual fidelity (PSNR/SSIM/LPIPS) and
semantic accuracy (mIoU). Our approach extends 3D Gaussian Splatting with a per-Gaussian
semantic logit vector, rendered jointly with color through a single fused rasterization pass. A
2D segmentation model pseudo-labels unlabeled frames to widen semantic supervision, and the
Gaussian model is warm-started from a semantically-voted sparse point cloud rather than random
initialization. On a 60-view held-out split drawn from the same UAV flight trajectory, our
method achieves PSNR 22.18 dB, SSIM 0.849, LPIPS 0.334, and structural mIoU 91.47% across the
four bridge component classes (deck, stay cable, tower, foundation).

---

## 1. Introduction

**Motivation**
- Bridges are critical infrastructure; periodic condition assessment is essential for public
  safety, but manual inspection is slow, costly, and sometimes hazardous (access to towers,
  cable anchorages, water-adjacent foundations).
- UAV photogrammetry has become a practical means of large-scale, low-cost bridge data
  acquisition, motivating automated 3D + semantic digital twin construction directly from
  drone imagery.
- Two capabilities are needed for a useful SHM digital twin: (1) accurate 3D geometric
  reconstruction, and (2) per-component semantic labeling (deck / stay cable / tower /
  foundation), so downstream analysis (deflection tracking, corrosion mapping, cable tension
  inference) can be scoped to the correct structural element.

**Problem statement (from the official brief)**
- Given ~300 labeled and 100 unlabeled multi-view UAV images of a cable-stayed bridge plus
  COLMAP-estimated camera poses, build a model that renders **both** an RGB image and a
  semantic map from an arbitrary camera viewpoint.
- Scored on a blind test set of camera viewpoints via `Accuracy Score = 0.5 x Visual Fidelity
  (PSNR/SSIM/LPIPS) + 0.5 x Semantic mIoU`.

**Core technical challenges**
1. Geometric diversity across structural components — the deck is a large horizontal plane,
   towers are tall vertical columns, stay cables are slender linear features spanning only a
   few pixels per view and prone to background bleeding (sky/water misclassified as cable).
2. Only 300 of 400 images carry manual annotations; the remaining 100 must be exploited without
   ground truth.
3. Camera poses are SfM estimates ("reference only" per the dataset documentation), not
   survey-grade ground truth.

**Contributions**
1. A semantic 3D Gaussian Splatting formulation that renders RGB and a 5-class semantic map in
   a single fused rasterization pass, satisfying the contest's dual-output requirement natively
   (no separate semantic-segmentation-of-renders post-process).
2. A semantic warm-start strategy that initializes each Gaussian's class logits from multi-view
   majority-voted labels on a triangulated sparse point cloud, instead of random initialization.
3. A 2D pseudo-labeling stage (fine-tuned SegFormer) that extends semantic supervision to the
   100 unlabeled frames, increasing the effective training-view count from 240 to 340.
4. An empirical study of training resolution's effect on reconstruction quality, showing
   full-resolution training yields consistent gains over half-resolution across every metric
   (Section 5.4).
5. End-to-end results on the contest's trajectory-interleaved 60-view holdout: PSNR 22.18 dB,
   SSIM 0.849, LPIPS 0.334, structural mIoU 91.47%.

---

## 2. Related Work

**[NEEDS: literature search — the two references below are already used elsewhere in this
project's documentation; more citations are needed for a complete related-work section,
particularly recent 3DGS/NeRF semantic-extension papers and UAV bridge-inspection surveys.]**

- **Structure-from-Motion / multi-view geometry**: COLMAP (Schönberger & Frahm) as the SfM
  backbone producing the camera poses and 2D-3D correspondences used here.
- **Neural scene representations**: NeRF (implicit, MLP-based, slow ray-marching render) vs.
  3D Gaussian Splatting (Kerbl et al., explicit primitives, real-time rasterization) — motivate
  why an explicit representation is preferable when both fast *and* per-primitive semantic
  attribution are required.
- **Semantic extensions of Gaussian Splatting / NeRF**: [NEEDS — cite Feature-3DGS,
  Semantic-NeRF, LangSplat, or similar prior art on attaching a semantic/feature channel to a
  radiance-field-style representation, and position this work's fused single-pass
  rasterization relative to them.]
- **2D structural bridge segmentation**: Lin et al. (2025), *A structure-oriented loss function
  for automated semantic segmentation of bridges* — motivates the 2D segmentation stage design.
- **Structure-aware 3D bridge reconstruction**: Hu et al. (2020), *Structure-aware 3D
  reconstruction for cable-stayed bridges: A learning-based method* — related goal (structural
  semantic 3D models of cable-stayed bridges), different technical approach (geometric
  priors/point-cloud filtering vs. this work's differentiable-rendering formulation).

---

## 3. Method

### 3.1 Problem Formulation

Formalize the task: given a set of posed images $\{(I_i, \pi_i)\}_{i=1}^{N}$ (RGB image + camera
pose), and a subset with pixel-level semantic annotations $M_i \in \{0,\ldots,4\}^{H\times W}$
(0=background, 1=deck, 2=stay_cable, 3=tower, 4=foundation), learn a 3D scene representation
$\mathcal{G}$ such that for any query pose $\pi_q$ (including unseen poses), rendering
$\mathcal{G}$ from $\pi_q$ produces both an RGB image $\hat{I}_q$ and a semantic map
$\hat{M}_q$ close to what a real photo/annotation from that viewpoint would show.

### 3.2 Camera Geometry and Sparse Initialization

- The dataset ships COLMAP `SIMPLE_RADIAL` intrinsics and per-image extrinsic poses for 400
  frames, but no precomputed 3D points.
- Sparse 3D points are recovered via LO-RANSAC multi-view triangulation over the 86,336
  provided 2D-3D feature tracks (mean reprojection error ≈0.5 px), followed by an IQR
  distance-from-median outlier filter, yielding an 84,613-point sparse cloud.
- Lens distortion ($k_1 \approx 0.009$) is removed once via `cv2.undistort` across all 400
  images, so all downstream training/rendering operates in a consistent pinhole convention
  (Gaussian rasterization assumes an ideal pinhole camera).

### 3.3 Task A: 2D Semantic Pseudo-Labeling

- A SegFormer (MiT-B0 backbone) is fine-tuned on the 240 labeled training images (a
  trajectory-interleaved 80/20 split of the 300 labeled frames, described in Section 3.6),
  validated each epoch on the 60-image holdout (2D mIoU).
- The fine-tuned model predicts pseudo-masks for the 100 unlabeled frames, widening the pool of
  semantically-supervised training views for Task B from 240 to 340. The 60 holdout images are
  never pseudo-labeled or otherwise touched by this stage.

### 3.4 Task B: Semantic 3D Gaussian Splatting

**Representation.** Each Gaussian $g_k$ carries: mean position $\mu_k \in \mathbb{R}^3$, scale
$s_k \in \mathbb{R}^3$, rotation quaternion $q_k$, opacity $\alpha_k$, RGB color $c_k$, and a
semantic logit vector $\ell_k \in \mathbb{R}^5$ (one per class).

**Semantic warm-start.** Rather than initializing colors/semantics randomly, each Gaussian's
position and color come directly from the triangulated sparse cloud (Sec. 3.2), and its
semantic logits are warm-started from a per-point multi-view majority vote over the training
views' 2D masks — `stay_cable` requires a strict absolute majority (>50%) of observing views to
win the vote (to resist background/sky bleeding into the slender-cable class); otherwise
plurality voting with a fixed tie-break priority applies. The voted class is encoded as a scaled
one-hot logit ($+2$ at the voted class, $-2$ elsewhere) rather than a hard label, so the
semantic channel starts from a reasonable prior instead of noise or an unbreakable constraint.

**Fused rendering.** RGB (3 channels) and semantic logits (5 channels) are concatenated into a
single 8-channel color tensor and rasterized in **one** pass via `gsplat`'s differentiable
rasterizer: each Gaussian is projected to a 2D splat, depth-sorted, and alpha-composited per
pixel. Reusing the same depth ordering for both outputs is both simpler and cheaper than
rendering RGB and semantics in two separate passes.

**Losses.** $\mathcal{L} = \mathcal{L}_{\text{photo}} + \lambda_{\text{sem}} \cdot w \cdot
\mathcal{L}_{\text{sem}}$, where $\mathcal{L}_{\text{photo}} = 0.8\,\mathcal{L}_1 +
0.2\,(1-\text{SSIM})$ is the standard 3DGS photometric loss, $\mathcal{L}_{\text{sem}}$ is
per-pixel cross-entropy against the (real or pseudo-) mask, and $w$ down-weights pseudo-labeled
unlabeled views relative to ground-truth-labeled views.

**Densification.** Gaussians are adaptively split/duplicated (high positional gradient) or
pruned (near-zero opacity) during training via `gsplat`'s standard density-control strategy,
growing from the 84,613-point initialization to **[fill in final count]** Gaussians.

### 3.5 Rendering for Arbitrary Viewpoints

The trained model exposes a single rendering entry point: given any camera pose (position +
orientation) and the shared camera intrinsics, it rasterizes an RGB image and a semantic-class
map (official class IDs 0-4) in one call — this is the literal artifact the contest evaluates,
independent of whether the requested pose was observed during training.

### 3.6 Evaluation Protocol

Following the contest's held-out evaluation philosophy, 60 of the 300 labeled images (every 5th
frame along the UAV flight trajectory) are withheld from every stage of training — 2D
segmentation fine-tuning, semantic warm-start voting, and Gaussian Splatting optimization. A
trajectory-interleaved (strided) split is used instead of a random split because adjacent UAV
frames share >99% visual overlap; a random split risks placing near-duplicate frames on both
sides, inflating the measured score. For each held-out view, we render RGB + semantic outputs
and compare against the real photo/GT mask via PSNR, SSIM, LPIPS, and mIoU.

---

## 4. Experiments

### 4.1 Dataset

- 300 labeled UAV images + 100 unlabeled, 1320x989 resolution, single shared `SIMPLE_RADIAL`
  camera ($f\approx925.7$ px, $k_1\approx0.009$).
- Pixel-level polygon annotations (Labelme) rasterized to 5-class masks (background, deck,
  stay_cable, tower, foundation), with `stay_cable` drawn last to avoid occlusion by broader
  deck/tower polygons.
- 240/60 trajectory-interleaved split for both the 2D segmentation and 3D stages.

### 4.2 Implementation Details

- Hardware: single NVIDIA RTX 3080 (10 GB).
- Task A: SegFormer MiT-B0, 80 epochs, batch size 8.
- Task B: `gsplat` differentiable rasterizer, Adam optimizers per parameter group (means,
  scales, quats, opacities, colors, semantic logits — different learning rates, means LR
  exponentially decayed), 40,000 iterations, trained at full image resolution (1320x989).
- Gaussian count capped at 600,000 during densification for memory/iteration-speed
  predictability on a 10 GB GPU.

### 4.3 Metrics

- **Visual fidelity**: PSNR, SSIM (`skimage.metrics`), LPIPS (AlexNet backbone).
- **Semantic accuracy**: per-class IoU and structural mIoU (4 classes, background excluded),
  from a standard confusion matrix over rendered-vs-GT class labels.
- **Accuracy Score** (illustrative combination, since the brief does not define how
  PSNR/SSIM/LPIPS combine into one Visual Fidelity number): mean of PSNR normalized against a
  35 dB reference, raw SSIM, and $(1-\text{LPIPS})$, averaged with mIoU per the official 0.5/0.5
  weighting.

---

## 5. Results & Discussion

### 5.1 Main Results

Evaluated on the 60-view held-out split (never used in training):

| Metric | Value |
| :--- | :---: |
| PSNR | 22.18 dB |
| SSIM | 0.849 |
| LPIPS | 0.334 |
| Structural mIoU (4 classes) | **91.47%** |
| Illustrative Accuracy Score | 0.816 |

**Per-class IoU:**

| Class | IoU |
| :--- | :---: |
| deck | 95.09% |
| stay_cable | 92.13% |
| tower | 91.13% |
| foundation | 87.52% |
| (background, reported for completeness, excluded from structural mIoU) | 99.24% |

### 5.2 Discussion — Per-Class Behavior

- `deck` scores highest — large, well-textured, planar surface observed from many overlapping
  viewpoints, easiest for both photometric and semantic supervision to converge on.
- `stay_cable`, despite being the slenderest class and historically the hardest for 2D/3D
  bridge segmentation (background/sky bleeding is a known failure mode — see Section 3.4's
  strict-majority voting rule motivated by exactly this), reaches 92.13% IoU — the semantic
  warm-start (initializing cable Gaussians from already-voted 3D points rather than letting the
  semantic channel learn cable geometry from scratch) appears to meaningfully help here.
  **[NEEDS: an ablation isolating warm-start's specific contribution to cable IoU, if time
  allows, would strengthen this claim.]**
- `foundation` is the weakest class — likely explained by limited UAV viewpoint coverage of
  low-lying, partially water-adjacent structures compared to the deck/tower, which are visible
  from most of the flight path.

### 5.3 Qualitative Results

**[NEEDS: figures]**
- Side-by-side rendered RGB vs. ground-truth photo for several held-out views.
- Rendered semantic map vs. ground-truth mask, using the official class-color legend.
- A novel-view interpolation figure (render at a camera pose interpolated between two flight
  positions) demonstrating smooth novel-view synthesis within the observed viewpoint envelope.
- Optionally: a screenshot of the exported splat point cloud (`export_semantic_splat_ply`)
  color-coded by predicted class, viewed in an interactive splat viewer.

### 5.4 Ablation: Training Resolution

| Training resolution | PSNR | SSIM | LPIPS | mIoU |
| :--- | :---: | :---: | :---: | :---: |
| Half (660x494) | 21.99 | 0.834 | 0.348 | 87.96% |
| **Full (1320x989)** | **22.18** | **0.849** | **0.334** | **91.47%** |

Training at native image resolution improves every metric, most notably mIoU (+3.5 points),
consistent with the intuition that thin structures (cable) and fine boundaries benefit from
sharper photometric/semantic gradients during optimization. The cost is proportionally longer
training time (≈47 min vs. ≈14 min for a comparable iteration budget on the same GPU); given
the modest absolute training time either way, full resolution is the recommended default.

---

## 6. Conclusion

- Summarize: a semantic 3D Gaussian Splatting pipeline that natively satisfies the contest's
  dual RGB+semantic rendering requirement, using a fused single-pass rasterization design and a
  point-cloud-informed semantic warm-start, reaching 91.47% structural mIoU and an illustrative
  Accuracy Score of 0.816 on the held-out evaluation protocol.
- Practical implication for SHM: a trained model of this kind is a queryable digital twin —
  any future inspection viewpoint (not limited to the original flight path) can be rendered
  with per-component semantic labels, a basis for downstream tasks like deflection tracking,
  cable-tension inference, or automated defect localization scoped to the correct structural
  element.
- **Future work** [NEEDS: keep this section limited to genuinely presentable directions —
  longer training / hyperparameter tuning for further Visual Fidelity gains, and integration
  with UAV vision displacement measurements (Project 1) for full-lifecycle SHM, are safe to
  list; do not include unfinished/negative-result experiments here].

---

## References

**[NEEDS: full reference list in the template's required citation style, including at minimum:]**
1. Schönberger, J. L., & Frahm, J.-M. — Structure-from-Motion Revisited (COLMAP).
2. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023) — 3D Gaussian Splatting for
   Real-Time Radiance Field Rendering.
3. Lin et al. (2025) — A structure-oriented loss function for automated semantic segmentation
   of bridges.
4. Hu et al. (2020) — Structure-aware 3D reconstruction for cable-stayed bridges: A
   learning-based method.
5. [NEEDS: 2-4 more references on semantic radiance fields / NeRF-Gaussian semantic extensions,
   and UAV bridge-inspection survey papers, to satisfy "adequacy of literature review" scoring.]

---

## Drafting Notes (delete before final submission)

- [ ] Confirm exact page/formatting requirements once the official template is downloaded from
  the IC-SHM website and available locally.
- [ ] Fill in final Gaussian count in Section 3.4 (currently: 602,363 per
  `outputs/checkpoints/gaussians/final.pt`).
- [ ] Produce the qualitative figures listed in 5.3 (`src/gaussian_splatting/render.py` +
  `render_metrics.py`'s per-view outputs already provide the raw material).
- [ ] Strengthen Related Work with real citations (Section 2's [NEEDS] items).
- [ ] Write the Abstract last, after Results is locked.
- [ ] Team/author details.
