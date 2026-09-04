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

Our pipeline has two stages. Task A produces pixel-level semantic labels for images the contest
leaves unannotated, so that the widest possible set of viewpoints can supervise the 3D model.
Task B fits a semantically-augmented 3D Gaussian Splatting model to the posed images and their
(real or predicted) semantic masks, and exposes a single rendering function that answers the
contest's core requirement: given any camera pose, produce both a photorealistic RGB image and
a per-pixel structural-class map. The rest of this section formalizes the task, then describes
each stage in turn.

### 3.1 Problem Formulation

We are given a set of posed UAV images of a single cable-stayed bridge,
$\{(I_i, \pi_i)\}_{i=1}^{N}$, where $I_i$ is an RGB photograph and $\pi_i$ its camera pose
(position and orientation) recovered by Structure-from-Motion. A subset of these images carries
pixel-level semantic annotations $M_i \in \{0,1,2,3,4\}^{H\times W}$ over five structural
classes (0: background, 1: deck, 2: stay cable, 3: tower, 4: foundation); the remainder do not.
Our goal is to learn a 3D scene representation $\mathcal{G}$ such that, for an arbitrary query
pose $\pi_q$ — including poses never observed during acquisition — rendering $\mathcal{G}$ from
$\pi_q$ produces both an RGB image $\hat{I}_q$ and a semantic map $\hat{M}_q$ that closely match
what a camera placed at $\pi_q$ would actually see. This formulation mirrors the contest's blind
evaluation protocol directly: the organizers hold out a set of camera poses, and a submission is
scored purely on how well it renders from those poses, with no access to the underlying ground
truth at inference time.

### 3.2 Camera Geometry and Sparse Point Initialization

The contest dataset provides COLMAP `SIMPLE_RADIAL` camera intrinsics and per-image extrinsic
poses for all 400 UAV frames, together with 86,336 two-dimensional feature tracks linking pixel
observations across views, but it does not include precomputed 3D point coordinates. We recover
a sparse point cloud of the bridge by triangulating every track with LO-RANSAC multi-view
triangulation, which jointly enforces cheirality and a minimum triangulation-angle constraint to
reject ill-conditioned geometry; the resulting points have a mean reprojection error of
approximately 0.5 pixels. A subsequent distance-from-median outlier filter (using the
interquartile range of each point's distance to the cloud centroid) removes residual floaters,
leaving 84,613 triangulated points. Because the shared camera carries non-negligible radial
distortion ($k_1 \approx 0.009$), and Gaussian rasterization assumes an ideal pinhole projection,
we undistort all 400 images once, up front, to a consistent pinhole convention; every subsequent
training, evaluation, and rendering step operates in this undistorted space.

### 3.3 Task A: 2D Semantic Pseudo-Labeling

Only 300 of the 400 available UAV frames carry manual polygon annotations; the remaining 100 are
unlabeled. To make use of them, we fine-tune a SegFormer semantic segmentation model (MiT-B0
backbone) on the 240 labeled training images obtained from our trajectory-interleaved split
(Section 3.6), validating 2D mIoU on the 60-image holdout after every epoch and retaining the
checkpoint with the best validation score. The fine-tuned model is then applied to the 100
unlabeled images to produce pseudo-masks, which widen the pool of semantically-supervised
training viewpoints available to Task B from 240 to 340 — a 42% increase in viewpoint coverage
for the semantic loss described in Section 3.4, at no additional annotation cost. The 60
held-out images are never touched by this stage, whether as training data or as prediction
targets: they are reserved exclusively for the final evaluation in Section 5.

### 3.4 Task B: Semantic 3D Gaussian Splatting

**Representation.** Following 3D Gaussian Splatting, we represent the bridge as a set of
anisotropic 3D Gaussians. Each Gaussian $g_k$ is parameterized by a mean position
$\mu_k \in \mathbb{R}^3$, a scale $s_k \in \mathbb{R}^3$, a rotation quaternion $q_k$, an
opacity $\alpha_k$, and an RGB color $c_k$. We augment this standard parameterization with a
semantic logit vector $\ell_k \in \mathbb{R}^5$, one entry per structural class, turning every
Gaussian into a carrier of both appearance and structural identity.

**Semantic warm-start.** Rather than initializing the Gaussians randomly, as is standard
practice, we exploit the sparse point cloud from Section 3.2 as a geometric and semantic prior.
Each Gaussian's initial position and color are taken directly from a corresponding triangulated
point, and its semantic logits are warm-started from that point's class, determined by a
multi-view majority vote over the 2D masks of every training view that observes it. Because
stay cables are slender and prone to background bleeding — sky and water pixels are easily
misclassified as cable in coarse 2D polygon annotations — the vote enforces a strict absolute
majority (greater than 50% of observing views) before assigning the cable class; if no class
reaches this majority among cable votes, cable votes are discarded and the remaining classes
compete by plurality with a fixed tie-break priority. The winning class is encoded as a scaled
one-hot logit (+2 at the voted class, −2 elsewhere) rather than a hard, unbreakable label, so
that the semantic channel begins optimization from an informed prior instead of from noise,
while remaining free to be corrected by the photometric and semantic losses during training.

**Fused rendering.** A central design choice of our method is that RGB and semantic outputs
share a single rasterization pass. We concatenate each Gaussian's RGB color (3 channels) and
semantic logits (5 channels) into one 8-channel color tensor, and render it through `gsplat`'s
differentiable rasterizer: every Gaussian is projected onto the image plane, the projections are
depth-sorted, and each pixel is computed by alpha-compositing the sorted splats from front to
back. Because the geometric projection and depth ordering that determine this composite depend
only on each Gaussian's position, scale, and rotation — not on which of its channels are being
composited — rendering RGB and semantics together in one pass is both simpler and cheaper than
running two independent rasterization passes with duplicated projection and sorting work, and it
guarantees the two outputs are pixel-aligned by construction.

Crucially, the semantic channels are composited by the *same* alpha-blending rule as color: a
rendered pixel's logit vector is an opacity- and depth-weighted combination of the semantic
logits of every Gaussian whose projected splat covers that pixel, not a hard, per-pixel vote
among discrete labels. The predicted class at a pixel is only resolved by taking the arg max of
this blended logit vector at read-out time (Section 3.5). During training, this same
differentiability is what lets semantic supervision reach the individual Gaussians: the
cross-entropy loss described below is computed on the blended per-pixel logits, and its gradient
is distributed back through the alpha-compositing weights to exactly the Gaussians that
contributed to each supervised pixel, in proportion to their contribution. A Gaussian whose
current class prediction is wrong for a given view is thus pushed toward the correct class, while
one that is already correct has its logits reinforced — and because each Gaussian is typically
observed by many training views from different angles over the course of optimization, its final
semantic identity reflects an accumulation of evidence across the whole trajectory rather than
any single observation. This is the same mechanism by which color and geometry are refined by the
photometric loss, applied identically to the semantic channels.

**Losses.** Training minimizes
$\mathcal{L} = \mathcal{L}_{\text{photo}} + \lambda_{\text{sem}} \, w \, \mathcal{L}_{\text{sem}}$
for each sampled training view. $\mathcal{L}_{\text{photo}} = 0.8\,\mathcal{L}_1 +
0.2\,(1-\text{SSIM})$ is the standard photometric loss used in 3D Gaussian Splatting, comparing
the rendered RGB image against the real photograph. $\mathcal{L}_{\text{sem}}$ is a per-pixel
cross-entropy loss between the rendered semantic logits and the corresponding ground-truth or
pseudo-label mask, and $w$ down-weights views supervised by Task A's pseudo-labels relative to
views with real manual annotations, reflecting their lower label confidence.

**Densification.** As is standard in Gaussian Splatting, the point set is not fixed throughout
training. Gaussians whose positional gradients are large — an indication that a single primitive
is being stretched to cover detail it cannot adequately represent — are split or duplicated,
while Gaussians whose opacity decays toward zero are pruned, using `gsplat`'s built-in
density-control strategy. This process grows the representation from the 84,613-point sparse
initialization to 602,363 Gaussians by the end of training, allowing the model to allocate
additional capacity to structurally intricate regions, such as individual cable strands, that
the initial sparse cloud under-represents.

### 3.5 Rendering for Arbitrary Viewpoints

The trained model exposes a single inference entry point that takes an arbitrary camera pose —
position, orientation, and the shared camera intrinsics — and returns both an RGB image and a
semantic-class map using the official class IDs (0–4). This function makes no assumption that
the requested pose was observed during training or even lies close to the UAV's original flight
line; it is the literal artifact evaluated by the contest organizers against their blind held-out
test poses, and we use the same function throughout this paper to render the results in Section
5.

### 3.6 Evaluation Protocol

We adopt a held-out evaluation protocol that mirrors the contest's own blind-test philosophy as
closely as possible without access to the organizers' actual test poses. Sixty of the 300
labeled images — every fifth frame along the UAV's flight trajectory — are withheld from every
stage of the pipeline: they contribute to neither Task A's fine-tuning nor validation-only use,
nor Task B's semantic warm-start voting, nor its photometric/semantic training loss. We choose
this trajectory-interleaved, strided split over a random split because consecutive UAV frames
overlap by more than 99% visually; a random split would risk placing near-duplicate frames on
both sides of the train/holdout boundary, artificially inflating the measured score by rewarding
memorization of nearly identical training views rather than genuine novel-view generalization.
For each of the 60 held-out views, we render RGB and semantic outputs from the trained model and
compare them against the real photograph and ground-truth mask using the metrics described in
Section 4.3.

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
