# Theoretical Foundations: SIFT, Epipolar Geometry, and COLMAP (SfM)

**Project**: IC-SHM 2026 (Project 2) — Structure-Aware 3D Semantic Point Cloud Reconstruction  
**Document Type**: Engineering Handbook & Theoretical Reference  
**Language**: English  

---

## 📑 Table of Contents
1. [Introduction: The Multi-View 3D Vision Problem](#1-introduction-the-multi-view-3d-vision-problem)
2. [Deep Dive into SIFT (Scale-Invariant Feature Transform)](#2-deep-dive-into-sift-scale-invariant-feature-transform)
   - [2.1 Why Sparse Features? (1.3M Pixels vs. ~3,000 Keypoints)](#21-why-sparse-features-13m-pixels-vs-3000-keypoints)
   - [2.2 Step 1: Scale-Space Extrema Detection (DoG)](#22-step-1-scale-space-extrema-detection-dog)
   - [2.3 Step 2: Keypoint Filtering & Localization (Contrast & Hessian Filters)](#23-step-2-keypoint-filtering--localization-contrast--hessian-filters)
   - [2.4 Step 3: Orientation Assignment (Rotation Invariance)](#24-step-3-orientation-assignment-rotation-invariance)
   - [2.5 Step 4: The 128-Dimensional Keypoint Descriptor](#25-step-4-the-128-dimensional-keypoint-descriptor)
   - [2.6 Step 5: Feature Matching & Lowe's Ratio Test](#26-step-5-feature-matching--lowes-ratio-test)
3. [Two-View Epipolar Geometry & Geometric Verification](#3-two-view-epipolar-geometry--geometric-verification)
4. [3D Triangulation: From 2D Pixels $(u, v)$ to 3D Space $(X, Y, Z)$](#4-3d-triangulation-from-2d-pixels-u-v-to-3d-space-x-y-z)
   - [4.1 Why a Single View Cannot Determine 3D Coordinates](#41-why-a-single-view-cannot-determine-3d-coordinates)
   - [4.2 Mathematical Formulation of DLT (Direct Linear Transform)](#42-mathematical-formulation-of-dlt-direct-linear-transform)
   - [4.3 Handling Skew Rays & Reprojection Error](#43-handling-skew-rays--reprojection-error)
5. [COLMAP Pipeline & Multi-View Structure-from-Motion (SfM)](#5-colmap-pipeline--multi-view-structure-from-motion-sfm)
   - [5.1 Sequential vs. Exhaustive Matching ($O(N)$ vs. $O(N^2)$)](#51-sequential-vs-exhaustive-matching-on-vs-on2)
   - [5.2 Bundle Adjustment (Non-linear Joint Optimization)](#52-bundle-adjustment-non-linear-joint-optimization)
6. [The Semantic Bridge: 2D Deep Learning Masks to 3D Digital Twin](#6-the-semantic-bridge-2d-deep-learning-masks-to-3d-digital-twin)

---

## 1. Introduction: The Multi-View 3D Vision Problem

Structure-from-Motion (SfM) solves the inverse problem of recovering **3D scene structure $(X, Y, Z)$** and **camera poses $(R, T)$** simultaneously from an unordered or sequential collection of 2D photographs.

```text
  UAV Image 1 (Frame #010)                           UAV Image 2 (Frame #015)
 ┌─────────────────────────┐                        ┌─────────────────────────┐
 │                         │                        │                         │
 │        • p₁(u₁, v₁)     │                        │               • p₂(u₂, v₂)
 │     (Cable anchor bolt) │                        │      (Cable anchor bolt)│
 └────────────┬────────────┘                        └────────────┬────────────┘
              │                                                  │
              ▼                                                  ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: LOCAL FEATURE EXTRACTION (SIFT)                                   │
 │ • Detect salient keypoints (corners, blobs, distinct anchors)              │
 │ • Compute 128-D rotation/scale-invariant descriptor vectors                │
 └─────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: TWO-VIEW FEATURE MATCHING & EPIPOLAR GEOMETRY (RANSAC)            │
 │ • Compare 128-D descriptors using Lowe's Ratio Test (d₁ / d₂ < 0.7)        │
 │ • Enforce Epipolar constraint (p₂ᵀ F p₁ = 0) to eliminate visual outliers  │
 └─────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3: MULTI-VIEW 3D TRIANGULATION & BUNDLE ADJUSTMENT                   │
 │ • Intersect optical sight rays from observing cameras (DLT / LO-RANSAC)    │
 │ • Compute 3D world coordinates X = (X, Y, Z) and minimize reprojection err │
 └─────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 4: 2D-TO-3D SEMANTIC FUSION & GEOMETRIC FILTERING (IC-SHM Project 2) │
 │ • Back-project 2D AI segmentation masks onto 3D points via Majority Voting │
 │ • Filter structural components using bridge geometric priors (PCA / tubes) │
 └────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep Dive into SIFT (Scale-Invariant Feature Transform)

Proposed by **David G. Lowe (1999, 2004)**, SIFT transforms image data into scale-invariant coordinates relative to local features.

### 2.1 Why Sparse Features? (1.3M Pixels vs. ~3,000 Keypoints)

A standard UAV image in our contest dataset has a resolution of $1320 \times 989 = \mathbf{1,305,480\text{ pixels}}$.

1. **Flat regions (sky, water, uniform concrete)**: Moving in any direction $(\Delta x, \Delta y)$ produces zero gradient change. These pixels carry no unique signature and cannot be matched across camera angles.
2. **Straight edges (slender stay cables, lane markings)**: Motion along the edge direction produces zero change (known as the *Aperture Problem*). Only 1D localization is possible.
3. **Corners, blobs, and structural anchors (anchor bolts, pylon apex, girder joints)**: Gradients change sharply in **all 2D directions**. These points provide unique, sub-pixel accurate $(x, y)$ localization.

SIFT selectively extracts approximately **2,000 to 5,000 sparse keypoints** per frame (~0.3% of all pixels).

---

### 2.2 Step 1: Scale-Space Extrema Detection (DoG)

To ensure scale invariance (recognizing a feature whether the drone is 5 meters or 50 meters away), SIFT searches for stable features across all possible scales using a continuous scale parameter $\sigma$.

1. **Scale-Space Representation**:
   $$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$
   where $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} e^{-(x^2 + y^2)/2\sigma^2}$ is a variable-scale Gaussian filter.

2. **Difference-of-Gaussians (DoG)**:
   DoG acts as an efficient approximation of the scale-normalized Laplacian-of-Gaussian $\sigma^2 \nabla^2 G$:
   $$D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y) = L(x, y, k\sigma) - L(x, y, \sigma)$$

```text
 Octave i+1          ┌──────────┐
 (Downsampled x2)    │   DoG    │  <-- 9 neighbor pixels at scale above
                     └──────────┘
                          ▲
 Octave i            ┌────┴─────┐
 (Current Scale)     │  • (x,y) │  <-- Compared against 8 neighbors at current scale
                     └────┬─────┘
                          ▼
                     ┌──────────┐
                     │   DoG    │  <-- 9 neighbor pixels at scale below
                     └──────────┘
```

3. **26-Neighbor Extrema Search**:
   A pixel at $(x, y, \sigma)$ is selected as a candidate keypoint **only if it is strictly larger (local maximum) or strictly smaller (local minimum) than all 26 neighboring pixels** (8 at current scale, 9 at scale above, 9 at scale below).

---

### 2.3 Step 2: Keypoint Filtering & Localization (Contrast & Hessian Filters)

Candidate extrema contain many unstable points vulnerable to noise or located along straight edges. SIFT applies **two rigorous mathematical rejection tests**:

#### A. Low-Contrast Rejection (Noise Filter)
Using a 2nd-order Taylor expansion around the candidate point $\mathbf{x} = (x, y, \sigma)^T$:
$$D(\mathbf{x}) = D + \frac{\partial D^T}{\partial \mathbf{x}} \mathbf{x} + \frac{1}{2} \mathbf{x}^T \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}$$

The sub-pixel peak location is computed via $\hat{\mathbf{x}} = - \left(\frac{\partial^2 D}{\partial \mathbf{x}^2}\right)^{-1} \frac{\partial D}{\partial \mathbf{x}}$.  
The value at the extremum is evaluated as:
$$|D(\hat{\mathbf{x}})| = \left| D + \frac{1}{2} \frac{\partial D^T}{\partial \mathbf{x}} \hat{\mathbf{x}} \right|$$
If $|D(\hat{\mathbf{x}})| < 0.03$, the point is **rejected as low-contrast noise**.

#### B. Edge Response Elimination (Hessian Eigenvalue Test)
Along a straight cable or line, the principal curvature across the edge is large, but along the edge it is near zero. To detect and discard edge points, SIFT computes the $2 \times 2$ Hessian matrix of spatial derivatives:
$$H = \begin{bmatrix} D_{xx} & D_{xy} \\ D_{xy} & D_{yy} \end{bmatrix}$$

Let $\alpha$ be the largest eigenvalue (curvature across edge) and $\beta$ be the smallest eigenvalue (curvature along edge), with ratio $r = \alpha / \beta$:
$$\text{Tr}(H) = D_{xx} + D_{yy} = \alpha + \beta, \quad \text{Det}(H) = D_{xx} D_{yy} - (D_{xy})^2 = \alpha \beta$$
$$\frac{\text{Tr}(H)^2}{\text{Det}(H)} = \frac{(\alpha + \beta)^2}{\alpha \beta} = \frac{(r + 1)^2}{r}$$

- **Edge Rejection Criterion**:
  $$\frac{\text{Tr}(H)^2}{\text{Det}(H)} < \frac{(r_{\text{threshold}} + 1)^2}{r_{\text{threshold}}} \quad (\text{typically } r_{\text{threshold}} = 10)$$
  If $r > 10$, the keypoint is located on an elongated edge and is **discarded**. Only well-defined 2D corners and blobs are retained.

---

### 2.4 Step 3: Orientation Assignment (Rotation Invariance)

To achieve rotational invariance, each keypoint is assigned a canonical local coordinate orientation:

1. For all sample points $L(x, y)$ in a Gaussian-weighted neighborhood around the keypoint at scale $\sigma$:
   - Gradient magnitude: $m(x, y) = \sqrt{(L(x+1, y) - L(x-1, y))^2 + (L(x, y+1) - L(x, y-1))^2}$
   - Gradient orientation: $\theta(x, y) = \text{atan2}(L(x, y+1) - L(x, y-1), L(x+1, y) - L(x, y-1))$
2. Build an orientation histogram with **36 bins** covering $360^\circ$ ($10^\circ$ per bin), weighted by $m(x, y)$ and a circular Gaussian window.
3. The dominant peak in the histogram defines the keypoint's **primary orientation $\theta$**. Any secondary peak within 80% of the dominant peak creates an additional keypoint with identical location but distinct orientation.

All subsequent descriptor measurements are rotated relative to $\theta$, ensuring complete rotational invariance.

---

### 2.5 Step 4: The 128-Dimensional Keypoint Descriptor

The descriptor encodes the local spatial layout of gradient orientations around the keypoint:

```text
 ┌─────────────────────────────────────────────────────────────┐
 │       16x16 PIXEL CANONICAL NEIGHBORHOOD (Rotated by θ)     │
 │                                                             │
 │   ┌──────┬──────┬──────┬──────┐                             │
 │   │ 8-dir│ 8-dir│ 8-dir│ 8-dir│  4 subregions (Row 1)       │
 │   ├──────┼──────┼──────┼──────┤                             │
 │   │ 8-dir│ 8-dir│ 8-dir│ 8-dir│  4 subregions (Row 2)       │
 │   ├──────┼──────┼──────┼──────┤  Total: 16 subregions       │
 │   │ 8-dir│ 8-dir│ 8-dir│ 8-dir│  x 8 orientation bins       │
 │   ├──────┼──────┼──────┼──────┤  = 128-dimensional vector   │
 │   │ 8-dir│ 8-dir│ 8-dir│ 8-dir│  4 subregions (Row 4)       │
 │   └──────┴──────┴──────┴──────┘                             │
 └─────────────────────────────────────────────────────────────┘
```

1. **Neighborhood Window**: A $16 \times 16$ pixel region around the keypoint is divided into a $4 \times 4$ array of subregions (each $4 \times 4$ pixels).
2. **Subregion Histogram**: In each $4 \times 4$ subregion, gradient orientations are accumulated into an 8-bin histogram (cardinal and diagonal directions: $0^\circ, 45^\circ, 90^\circ, \dots, 315^\circ$).
3. **Vector Assembly**: Concatenating all histograms produces a feature vector of length:
   $$4 \times 4 \times 8 = \mathbf{128\text{ dimensions}}$$
4. **Illumination Normalization**:
   - The 128-D vector is normalized to unit length: $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|_2$ (cancels linear brightness changes).
   - Values are clipped at a maximum of $0.2$ and re-normalized (cancels non-linear camera sensor saturation effects).

---

### 2.6 Step 5: Feature Matching & Lowe's Ratio Test

When comparing keypoints between two images $I_A$ and $I_B$:

For a descriptor $\mathbf{d}_A \in I_A$, find:
- $\mathbf{d}_{B, 1}$: Closest neighbor in $I_B$ with Euclidean distance $d_1 = \|\mathbf{d}_A - \mathbf{d}_{B, 1}\|_2$.
- $\mathbf{d}_{B, 2}$: Second-closest neighbor in $I_B$ with Euclidean distance $d_2 = \|\mathbf{d}_A - \mathbf{d}_{B, 2}\|_2$.

**Lowe's Ratio Criterion**:
$$\frac{d_1}{d_2} < \tau \quad (\text{typically } \tau = 0.70 \text{ or } 0.80)$$

- **Rationale**: If the feature is distinctive, the true match will be much closer than any accidental background similarity ($d_1 \ll d_2$). If the feature is repetitive (e.g., identical window panes, uniform road texture), $d_1 \approx d_2$, and the ambiguous match is rejected.

---

## 3. Two-View Epipolar Geometry & Geometric Verification

Even after the ratio test, visual matches may be physically incorrect due to repeated bridge patterns. COLMAP enforces **Epipolar Geometry** to verify physical consistency.

```text
         Camera Center C₁                    Camera Center C₂
                \                                  /
                 \                                /
                  \  Baseline Vector b = C₂ - C₁ /
                   \                            /
                    ▼                          ▼
               Epipole e₁                  Epipole e₂
                    \                          /
                     \                        /  Epipolar Line l₂
                      \                      /
                       \      Point X       /
                        \     (3D Space)   /
                         \        •       /
                          \      / \     /
                           \    /   \   /
                            ▼  ▼     ▼ ▼
                           p₁(u₁,v₁)  p₂(u₂,v₂)
                           [Image 1]  [Image 2]
```

### Mathematical Formulation
1. Any 3D point $\mathbf{X}$, the two camera optical centers $\mathbf{C}_1, \mathbf{C}_2$, and the projected image points $\mathbf{p}_1, \mathbf{p}_2$ lie on a single 3D plane (the **Epipolar Plane**).
2. The intersection of the epipolar plane with the second image plane forms the **Epipolar Line** $\mathbf{l}_2$.
3. **Fundamental Matrix Constraint**:
   $$\mathbf{p}_2^T \mathbf{F} \mathbf{p}_1 = 0$$
   where $\mathbf{F} = \mathbf{K}_2^{-T} [\mathbf{t}]_\times \mathbf{R} \mathbf{K}_1^{-1}$ is the $3 \times 3$ Fundamental Matrix of rank 2.

### RANSAC Verification
COLMAP runs **RANSAC (Random Sample Consensus)** with the 8-Point or 5-Point algorithm:
- Samples minimal subsets of 8 correspondence pairs to estimate candidate matrices $\mathbf{F}$.
- Computes Sampson distance / symmetric epipolar transfer error for all matches.
- Matches with distance $> 1.0\text{ px}$ from the epipolar line are **permanently removed as outliers**.

---

## 4. 3D Triangulation: From 2D Pixels $(u, v)$ to 3D Space $(X, Y, Z)$

### 4.1 Why a Single View Cannot Determine 3D Coordinates

A single 2D pixel observation $(u, v)$ defines an optical ray in camera space extending infinitely along depth $Z$:
$$\mathbf{p} = \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \mathbf{K} [\mathbf{R} | \mathbf{T}] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$
Because depth along the line of sight is lost in perspective projection, determining $(X, Y, Z)$ requires **at least two independent camera views with non-zero baseline**.

---

### 4.2 Mathematical Formulation of DLT (Direct Linear Transform)

Let a 3D point in homogeneous coordinates be $\mathbf{X} = [X, Y, Z, 1]^T$, and let the $3 \times 4$ camera projection matrix for view $k$ be $\mathbf{P}_k = \mathbf{K} [\mathbf{R}_k | \mathbf{T}_k]$, with rows $\mathbf{p}_k^{1T}, \mathbf{p}_k^{2T}, \mathbf{p}_k^{3T}$.

The 2D projected coordinates are $u_k = \frac{\mathbf{p}_k^{1T} \mathbf{X}}{\mathbf{p}_k^{3T} \mathbf{X}}$ and $v_k = \frac{\mathbf{p}_k^{2T} \mathbf{X}}{\mathbf{p}_k^{3T} \mathbf{X}}$.

Cross-multiplying yields two linearly independent equations per observing view:
$$u_k (\mathbf{p}_k^{3T} \mathbf{X}) - (\mathbf{p}_k^{1T} \mathbf{X}) = 0$$
$$v_k (\mathbf{p}_k^{3T} \mathbf{X}) - (\mathbf{p}_k^{2T} \mathbf{X}) = 0$$

For $N \ge 2$ camera views, this forms an overdetermined linear system:
$$\mathbf{A} \mathbf{X} = \mathbf{0}, \quad \text{where } \mathbf{A} \in \mathbb{R}^{2N \times 4}$$

**Solution via Singular Value Decomposition (SVD)**:
$$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
The optimal 3D coordinate $\mathbf{X}$ is the singular vector corresponding to the smallest singular value (the last column of $\mathbf{V}$). De-homogenizing gives:
$$X = V_{1,4} / V_{4,4}, \quad Y = V_{2,4} / V_{4,4}, \quad Z = V_{3,4} / V_{4,4}$$

---

### 4.3 Handling Skew Rays & Reprojection Error

In practice, due to lens distortion and sensor noise, optical rays do not intersect perfectly in 3D (forming *skew lines*).

```text
       Ray 1 (from Camera 1)
       ───────────────────────────────\
                                       \
                                        \     δ (Closest distance / Skew gap)
                                         │  <-- Solved via LO-RANSAC
                                        /
       ───────────────────────────────/
       Ray 2 (from Camera 2)
```

The geometric quality of a triangulated 3D point $\mathbf{X}_i$ is quantified by its **Mean Reprojection Error**:
$$\epsilon_i = \frac{1}{|V_i|} \sum_{k \in V_i} \left\| \begin{bmatrix} u_{ik} \\ v_{ik} \end{bmatrix} - \pi(\mathbf{K}, \mathbf{R}_k, \mathbf{T}_k, \mathbf{X}_i) \right\|_2$$
where $\pi(\cdot)$ is the projection function and $V_i$ is the set of cameras observing point $i$.  
- In our pipeline, points with $\epsilon_i > 1.0\text{ px}$ or small triangulation angles ($< 2.0^\circ$) are pruned.

---

## 5. COLMAP Pipeline & Multi-View Structure-from-Motion (SfM)

### 5.1 Sequential vs. Exhaustive Matching ($O(N)$ vs. $O(N^2)$)

Given $N = 400$ drone images:
- **Exhaustive Matching**: Tests all pairs: $\frac{N(N-1)}{2} = \frac{400 \times 399}{2} = \mathbf{79,800\text{ pairs}}$. At ~3,000 features per image, this requires computing billions of vector dot products.
- **Sequential Matching**: Exploits the continuous UAV flight trajectory. Frame $i$ only shares visual overlap with neighboring frames $i \pm K$ (e.g., $K = 20$).
  $$\text{Total Pairs} \approx N \times K = 400 \times 20 = \mathbf{8,000\text{ pairs}} \quad (\approx 10\times\text{ faster})$$

COLMAP's `pycolmap.match_sequential` performs this matching on the GPU (NVIDIA RTX 3080) in under 60 seconds.

---

### 5.2 Bundle Adjustment (Non-linear Joint Optimization)

After initial triangulation, camera parameters and 3D point coordinates accumulate slight drift. **Bundle Adjustment (BA)** performs global non-linear least squares optimization using the **Levenberg-Marquardt (LM)** algorithm:

$$\min_{\{\mathbf{R}_k, \mathbf{T}_k, \mathbf{K}\}, \{\mathbf{X}_i\}} \sum_{i} \sum_{k \in V_i} \rho\left( \left\| \mathbf{x}_{ik} - \pi(\mathbf{K}, \mathbf{R}_k, \mathbf{T}_k, \mathbf{X}_i) \right\|^2 \right)$$
where $\rho(\cdot)$ is a robust Cauchy loss function to attenuate outlier residuals.

---

## 6. The Semantic Bridge: 2D Deep Learning Masks to 3D Digital Twin

COLMAP reconstructs purely geometric 3D points $(X, Y, Z)$ without semantic class identities. The bridge to a **Semantic 3D Digital Twin** works as follows:

```text
                          3D Point X_i = (X, Y, Z)
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
    Observing Cam #10         Observing Cam #15         Observing Cam #22
  Project to (u₁₀, v₁₀)     Project to (u₁₅, v₁₅)     Project to (u₂₂, v₂₂)
           │                         │                         │
           ▼                         ▼                         ▼
  Query 2D Mask #10         Query 2D Mask #15         Query 2D Mask #22
  (AI: "stay_cable")        (AI: "stay_cable")        (AI: "deck" - Occluded)
           │                         │                         │
           └─────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
                      MULTI-VIEW MAJORITY VOTING
                 • stay_cable votes: 2/3 (66.7% > 50%)
                 • deck votes:       1/3 (33.3%)
                                     │
                                     ▼
                 Final Label for X_i = 2 (stay_cable)
                 RGB Color = (0, 255, 255) Cyan
```

### Multi-View Majority Voting Rules (in `src/reconstruction/semantic_projector.py`):
1. **Slender Structure Safeguard**: Due to thin cable boundaries, a 3D point is assigned to `stay_cable` (Class 2) **only if it achieves strict absolute majority ($> 50\%$)** across observing views.
2. **Plurality Voting**: If no absolute majority exists for cables, cable votes are discarded and plurality voting applies across remaining structural classes (`tower`, `deck`, `foundation`).
3. **Geometric Prior Filtering**: Post-fusion, structural filters (`filter_deck_plane`, `filter_tower_core`, `filter_cable_tower_planes`) remove remaining semantic boundary bleed and multi-path reflection noise.

---

## 📚 Key References
1. **Lowe, D. G. (2004)**. *Distinctive Image Features from Scale-Invariant Keypoints*. International Journal of Computer Vision (IJCV), 60(2), 91–110.
2. **Hartley, R., & Zisserman, A. (2004)**. *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
3. **Schönberger, J. L., & Frahm, J. M. (2016)**. *Structure-from-Motion Revisited*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
4. **Lin, Y., et al. (2025)**. *A structure-oriented loss function for automated semantic segmentation of bridges*. Computer-Aided Civil and Infrastructure Engineering.
5. **Hu, G., et al. (2020)**. *Structure-aware 3D reconstruction for cable-stayed bridges: A learning-based method*. Computer-Aided Civil and Infrastructure Engineering.
