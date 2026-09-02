# IC-SHM 2026 (Project 2) — Evaluation Framework & Performance Metrics

This document outlines the comprehensive evaluation methodology for the **Structure-Aware 3D Semantic Point Cloud Reconstruction of Cable-Stayed Bridges**. The evaluation protocol integrates **3D Semantic Segmentation Metrics**, **3D Geometric Reconstruction Accuracy**, and **Domain-Specific Structural Health Monitoring (SHM) Constraints**.

> **Research Alignment**: The selected metrics explicitly draw upon standards established in recent research:
> - **Lin et al. (2025)**: Inspired our 2D/3D semantic metrics (mIoU, OA) and the explicit need to track slender elements like cables ($IoU_{\text{cable}}$).
> - **Hu et al. (2020)**: Inspired the integration of geometric metrics (Reprojection Error, Chamfer Distance) alongside domain-specific structural constraints (Deck Planarity MAD, Cable Fan Deviation).

---

## 📑 Complete Metrics Taxonomy & Rationale

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     IC-SHM 2026 EVALUATION TAXONOMY (3 PILLARS)             │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ PILLAR 1: 3D Semantic Classification Metrics (mIoU_struct, mIoU_all, OA, F1)│
 │   ➔ Evaluates categorical labeling accuracy across the 4 bridge components  │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ PILLAR 2: Photogrammetric & Geometric Precision (Reprojection Error, Sim(3))│
 │   ➔ Evaluates 3D spatial fidelity, scale alignment, and camera accuracy     │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ PILLAR 3: Domain-Specific Structural Health Monitoring (SHM) Priors         │
 │   ➔ Evaluates adherence to civil engineering physics (Deck MAD, Cable Fan)  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Pillar 1: 3D Semantic Segmentation Metrics

Evaluates how accurately each 3D point $(X, Y, Z)$ is categorized into its corresponding structural class:
`0: background`, `1: deck`, `2: stay_cable`, `3: tower`, `4: foundation`.

### 1.1 Mean Intersection over Union ($mIoU$) — *Primary Metric*
$$IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}$$

To maintain rigorous mathematical clarity, we explicitly distinguish between two formulations:
1. **Structural $mIoU$ ($mIoU_{\text{struct}}$ — Primary Evaluation Benchmark)**:
   $$mIoU_{\text{struct}} = \frac{1}{4} \sum_{c=1}^{4} IoU_c = \frac{IoU_{\text{deck}} + IoU_{\text{stay\_cable}} + IoU_{\text{tower}} + IoU_{\text{foundation}}}{4}$$
   *Explicit Rationale*: The `background` class (Class 0) contains expansive sky and river regions. Including background in $mIoU$ artificially inflates or dilutes performance on structural components. Therefore, **$mIoU_{\text{struct}}$ is computed strictly over the 4 physical bridge classes ($C_{\text{struct}} = 4$)**.
2. **Global $mIoU$ ($mIoU_{\text{all}}$ — All 5 Classes)**:
   $$mIoU_{\text{all}} = \frac{1}{5} \sum_{c=0}^{4} IoU_c$$

- **Target**: $mIoU_{\text{struct}} > \mathbf{85.0\%}$.

---

### 1.2 Stay-Cable IoU ($IoU_{\text{cable}}$) — *Key Benchmark for Slender Structures*
$$IoU_{\text{cable}} = \frac{TP_{\text{cable}}}{TP_{\text{cable}} + FP_{\text{cable}} + FN_{\text{cable}}}$$

- **Why It Is Needed**: Stay cables represent slender linear features occupying $< 10\%$ of total points. Because 2D annotations cover whole cable regions rather than individual strands, background sky and water pixels easily leak into cable masks. $IoU_{\text{cable}}$ directly measures the model's resistance to background label bleeding.
- **Target**: $IoU_{\text{cable}} > \mathbf{75.0\%}$.

---

### 1.3 Overall Accuracy (OA) & Class Accuracy (mAcc)
$$OA = \frac{\sum_{c=0}^4 TP_c}{N_{\text{total}}}, \quad mAcc = \frac{1}{4} \sum_{c=1}^4 \frac{TP_c}{TP_c + FN_c}$$

---

### 1.4 Per-Class Precision, Recall, and $F_1$-Score
$$Precision_c = \frac{TP_c}{TP_c + FP_c}, \quad Recall_c = \frac{TP_c}{TP_c + FN_c}, \quad F_1^{(c)} = \frac{2 \cdot Precision_c \cdot Recall_c}{Precision_c + Recall_c}$$

---

## 2. Pillar 2: 3D Geometric Reconstruction Accuracy & Scale Recovery

### 2.1 Scale Ambiguity Resolution (Metric Scale Recovery via $\text{Sim}(3)$)

> ⚠️ **Methodological Note on Scale Ambiguity**: Monocular SfM (COLMAP) reconstructs scenes up to an arbitrary global scale factor $s \in \mathbb{R}^+$. To ensure physical distance metrics (measured in meters) are mathematically rigorous:
> 1. **Metric Scaling Factor $s$**: Computed by matching a known physical bridge dimension (e.g., standard deck width $W_{\text{deck}} = 12.0\text{ m}$ or tower height $H_{\text{tower}}$) against the reconstructed model:
>    $$s = \frac{L_{\text{true}}}{L_{\text{SfM}}}$$
> 2. **Sim(3) Umeyama Alignment**: When reference CAD/BIM or LiDAR coordinates $\mathbf{G}$ are available, the predicted cloud $\mathbf{P}$ is aligned via a 7-DOF transformation $[\mathbf{R}, \mathbf{t}, s]$:
>    $$\min_{s, \mathbf{R}, \mathbf{t}} \sum_{i} \|\mathbf{g}_i - (s \mathbf{R} \mathbf{p}_i + \mathbf{t})\|_2^2$$

All spatial metrics below are evaluated **after $\text{Sim}(3)$ metric scale alignment**.

---

### 2.2 Mean Reprojection Error ($e_{\text{reproj}}$)
$$e_{\text{reproj}} = \frac{1}{N} \sum_{i=1}^N \| p_{2D}^{(i)} - \pi(K, R_k, T_k, X_{3D}^{(i)}) \|_2$$

- **Why It Is Needed**: Measures optical ray convergence consistency independent of metric scale.
- **Target**: $e_{\text{reproj}} < \mathbf{1.0\text{ pixel}}$ (Ideal: $0.4 - 0.7\text{ px}$).

---

### 2.3 Spatial Point Density & Surface Coverage (Replacing Arbitrary Point Counts)

Rather than evaluating an arbitrary absolute point count ($> 75,000\text{ pts}$), which is sensitive to image resolution and SIFT thresholds, we define scale-invariant completeness metrics:

1. **Spatial Surface Density ($\rho_{\text{surface}}$)**:
   $$\rho_{\text{surface}} = \frac{N_{\text{structural}}}{\text{Estimated Surface Area } A_{\text{bridge}}} \quad (\text{Target: } \rho_{\text{surface}} \ge \mathbf{50\text{ pts/m}^2})$$
2. **Camera Registration Ratio ($R_{\text{reg}}$)**:
   $$R_{\text{reg}} = \frac{N_{\text{registered\_frames}}}{N_{\text{total\_frames}}} = \frac{400}{400} = \mathbf{100.0\%}$$
3. **Surface Coverage Completeness ($F\text{-score} @ 5\text{cm}$)**:
   Percentage of the bridge envelope covered by reconstructed points within $5\text{ cm}$ of the structural shell. (Target: $F(5\text{cm}) > \mathbf{90.0\%}$).

---

## 3. Pillar 3: Domain-Specific Structural Health Monitoring (SHM) Metrics

Evaluates physical, mechanical, and architectural adherence to cable-stayed bridge design priors (Hu et al. 2020).

### 3.1 Deck Planarity Residual MAD ($\text{MAD}_{\text{deck}}$)
$$\text{MAD}_{\text{deck}} = \text{median}\Big( \big| \mathbf{n}_{\text{deck}} \cdot \mathbf{x}_i + d_{\text{deck}} \big| \Big)$$

- **Why It Is Needed**: Civil bridge decks are continuous planar surfaces. SfM depth drift often warps flat roadways into curved or noisy surfaces. The 2-pass PCA plane residual MAD quantifies roadway surface roughness and noise.
- **Target**: $\text{MAD}_{\text{deck}} < \mathbf{0.05\text{ m}}$ ($5\text{ cm}$).

---

### 3.2 Cable Fan Plane Deviation ($\overline{\text{Deviation}}_{\text{cable}}$)
$$\overline{\text{Deviation}}_{\text{cable}} = \frac{1}{|P_{\text{cable}}|} \sum_{x \in P_{\text{cable}}} \min\Big( \big|(\mathbf{x} - \mathbf{x}_0) \cdot \mathbf{w} - d_{\text{left}}\big|, \big|(\mathbf{x} - \mathbf{x}_0) \cdot \mathbf{w} - d_{\text{right}}\big| \Big)$$

- **Why It Is Needed**: Cable stays are anchored along two symmetric vertical/inclined planes (Left & Right Fan Sheets). Because 2D region masks leak sky/river pixels into the cable class, points scatter across 3D space. This metric measures how tightly 3D cable points adhere to the true physical cable fan sheets.
- **Target**: $\overline{\text{Deviation}}_{\text{cable}} < \mathbf{0.10\text{ m}}$ ($10\text{ cm}$).

---

### 3.3 Off-Fan Cable Outlier Ratio & $\tau$ Tolerance Setting

$$\text{Outlier Ratio}(\tau) = \frac{\sum_{x \in P_{\text{cable}}} \mathbb{I}\big[\text{dist}(x, \text{Fan Planes}) > \tau\big]}{|P_{\text{cable}}|} \times 100\%$$

> ⚙️ **Parameter Setting for $\tau$**:
> - The threshold $\tau$ must account for physical cable bundle diameter ($D_{\text{cable}} \approx 10 - 20\text{ cm}$) plus triangulation tolerance:
>   $$\tau = \frac{D_{\text{cable}}}{2} + \epsilon_{\text{tol}} \approx 0.10\text{ m} \text{ to } 0.15\text{ m}$$
> - In our benchmark reports, we provide **Sensitivity Curves** across $\tau \in [0.05\text{ m}, 0.20\text{ m}]$ to demonstrate robustness across cable diameters.
> - **Target**: $\text{Outlier Ratio}(\tau = 0.10\text{m}) < \mathbf{2.0\%}$.

---

## 4. Master Benchmark Summary Scorecard

| Pillar | Metric | Mathematical Formulation | Recommended Benchmark Target | Rigorous Methodological Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **Pillar 1: Semantics** | **Structural $mIoU$** | $\frac{1}{4}\sum_{c=1}^4 IoU_c$ | **$> 85.0\%$** | Evaluated on the 4 bridge classes, explicitly excluding background |
| | **Cable $IoU$ ($IoU_{\text{cable}}$)** | $TP / (TP + FP + FN)$ | **$> 75.0\%$** | Measures resistance to background bleeding on slender cables |
| | **Overall Accuracy ($OA$)** | $\sum TP / N_{\text{total}}$ | **$> 92.0\%$** | Macro point classification accuracy across all 5 classes |
| **Pillar 2: Geometry** | **Mean Reprojection Error** | $\frac{1}{N}\sum \|p_{2D} - \pi(X_{3D})\|$ | **$< 1.0\text{ px}$** | Scale-invariant optical ray intersection accuracy |
| | **Spatial Point Density** | $N_{\text{struct}} / A_{\text{bridge}}$ | **$\ge 50\text{ pts/m}^2$** | Scale-independent density metric replacing raw point counts |
| | **Camera Registration Ratio** | $N_{\text{reg}} / N_{\text{total}}$ | **$100.0\%$ (400/400)** | Full coverage of all flight lines with zero missing frames |
| **Pillar 3: SHM Priors** | **Deck Planarity MAD** | $\text{median}(\|n \cdot x + d\|)$ | **$< 0.05\text{ m}$** | Measured after metric scale recovery via $\text{Sim}(3)$ |
| | **Cable Fan Deviation** | $\text{mean}(\text{dist}(x, \text{Fan Planes}))$ | **$< 0.10\text{ m}$** | Physical alignment of cable points onto left/right fan sheets |
| | **Off-Fan Outlier Ratio** | $\text{Ratio}(\text{dist} > \tau)$ | **$< 2.0\%$** | Evaluated at $\tau = 0.10\text{ m}$ with sensitivity analysis across $[0.05, 0.20\text{m}]$ |
