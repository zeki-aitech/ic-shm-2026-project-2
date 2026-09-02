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
 │ PILLAR 1: 3D Semantic Classification Metrics (IoU, mIoU, OA, F1)            │
 │   ➔ Evaluates categorical labeling accuracy across the 4 bridge components  │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ PILLAR 2: Photogrammetric & Geometric Precision (Reprojection Error, CD)    │
 │   ➔ Evaluates 3D spatial fidelity and camera alignment precision            │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ PILLAR 3: Domain-Specific Structural Health Monitoring (SHM) Priors         │
 │   ➔ Evaluates adherence to civil engineering physics (Deck MAD, Cable Fan)  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Pillar 1: 3D Semantic Segmentation Metrics

Evaluates how accurately each 3D point $(X, Y, Z)$ is categorized into its corresponding structural class:
`0: background`, `1: deck`, `2: stay_cable`, `3: tower`, `4: foundation`.

### 1.1 Mean Intersection over Union (mIoU) — *Primary Metric*
$$IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \quad mIoU = \frac{1}{C_{\text{struct}}} \sum_{c=1}^{4} IoU_c$$

- **Why It Is Needed**: Standard accuracy ($OA$) is misleading on imbalanced data because massive surfaces (like the bridge deck) dominate the point count. $mIoU$ penalizes false positives and false negatives equally across all classes, ensuring small or rare classes are evaluated fairly.
- **Target**: $mIoU > \mathbf{85.0\%}$.

---

### 1.2 Stay-Cable IoU ($IoU_{\text{cable}}$) — *Key Benchmark for Slender Structures*
$$IoU_{\text{cable}} = \frac{TP_{\text{cable}}}{TP_{\text{cable}} + FP_{\text{cable}} + FN_{\text{cable}}}$$

- **Why It Is Needed**: Stay cables represent slender linear features occupying $< 10\%$ of total points. Because 2D annotations cover whole cable regions rather than individual strands, background sky and water pixels easily leak into cable masks. $IoU_{\text{cable}}$ directly measures the model's resistance to background label bleeding.
- **Target**: $IoU_{\text{cable}} > \mathbf{75.0\%}$.

---

### 1.3 Overall Accuracy (OA) & Class Accuracy (mAcc)
$$OA = \frac{\sum_{c} TP_c}{N_{\text{total}}}, \quad mAcc = \frac{1}{C} \sum_{c} \frac{TP_c}{TP_c + FN_c}$$

- **Why It Is Needed**: Provides a global macro-level summary of total correct classifications across the entire reconstructed point cloud.
- **Target**: $OA > \mathbf{92.0\%}$.

---

### 1.4 Per-Class Precision, Recall, and $F_1$-Score
$$Precision_c = \frac{TP_c}{TP_c + FP_c}, \quad Recall_c = \frac{TP_c}{TP_c + FN_c}, \quad F_1^{(c)} = \frac{2 \cdot Precision_c \cdot Recall_c}{Precision_c + Recall_c}$$

- **Why It Is Needed**: Helps diagnose whether a component is being over-predicted (low precision / high false alarms) or missed (low recall / under-segmentation).

---

## 2. Pillar 2: 3D Geometric Reconstruction Accuracy

Evaluates spatial fidelity, geometric precision, and noise suppression in the reconstructed 3D point cloud coordinates $(X, Y, Z)$.

### 2.1 Mean Reprojection Error ($e_{\text{reproj}}$)
$$e_{\text{reproj}} = \frac{1}{N} \sum_{i=1}^N \| p_{2D}^{(i)} - \pi(K, R_k, T_k, X_{3D}^{(i)}) \|_2$$

- **Why It Is Needed**: Measures optical consistency between 3D points and 2D sensor observations across all observing UAV cameras. A high reprojection error indicates camera pose drift or incorrect feature matching.
- **Target**: $e_{\text{reproj}} < \mathbf{1.0\text{ pixel}}$ (Ideal: $0.4 - 0.7\text{ px}$).

---

### 2.2 Chamfer Distance (CD) & F-Score @ 5cm
$$d_{CD}(P, G) = \frac{1}{|P|} \sum_{x \in P} \min_{y \in G} \|x - y\|_2^2 + \frac{1}{|G|} \sum_{y \in G} \min_{x \in P} \|x - y\|_2^2$$

- **Why It Is Needed**: Measures Euclidean deviation between the reconstructed point cloud and ground-truth as-designed CAD/BIM or LiDAR scans.
- **Target**: $d_{CD} < \mathbf{0.02\text{ m}^2}$, $F(5\text{cm}) > \mathbf{90.0\%}$.

---

## 3. Pillar 3: Domain-Specific Structural Health Monitoring (SHM) Metrics

Evaluates physical, mechanical, and architectural adherence to cable-stayed bridge design priors (Hu et al. 2020).

### 3.1 Deck Planarity Residual MAD ($\text{MAD}_{\text{deck}}$)
$$\text{MAD}_{\text{deck}} = \text{median}\Big( \big| \mathbf{n}_{\text{deck}} \cdot \mathbf{x}_i + d_{\text{deck}} \big| \Big)$$

- **Why It Is Needed**: Civil bridge decks are engineered as continuous planar surfaces. SfM depth drift often warps flat roadways into curved or noisy surfaces. The 2-pass PCA plane residual MAD quantifies roadway surface roughness and noise.
- **Target**: $\text{MAD}_{\text{deck}} < \mathbf{0.05\text{ m}}$ ($5\text{ cm}$).

---

### 3.2 Cable Fan Plane Deviation ($\overline{\text{Deviation}}_{\text{cable}}$)
$$\overline{\text{Deviation}}_{\text{cable}} = \frac{1}{|P_{\text{cable}}|} \sum_{x \in P_{\text{cable}}} \min\Big( \big|(\mathbf{x} - \mathbf{x}_0) \cdot \mathbf{w} - d_{\text{left}}\big|, \big|(\mathbf{x} - \mathbf{x}_0) \cdot \mathbf{w} - d_{\text{right}}\big| \Big)$$

- **Why It Is Needed**: Cable stays are anchored along two symmetric vertical/inclined planes (Left & Right Fan Sheets). Because 2D region masks leak sky/river pixels into the cable class, points scatter across 3D space. This metric measures how tightly 3D cable points adhere to the true physical cable fan sheets.
- **Target**: $\overline{\text{Deviation}}_{\text{cable}} < \mathbf{0.10\text{ m}}$ ($10\text{ cm}$).

---

### 3.3 Off-Fan Cable Outlier Ratio ($\text{Outlier Ratio}_{\text{cable}}$)
$$\text{Outlier Ratio} = \frac{\sum_{x \in P_{\text{cable}}} \mathbb{I}\big[\text{dist}(x, \text{Fan Planes}) > \tau\big]}{|P_{\text{cable}}|} \times 100\% \quad (\tau = 0.10\text{ m})$$

- **Why It Is Needed**: Quantifies the percentage of floating "ghost" cable artifacts caused by 2D background leakage.
- **Target**: $\text{Outlier Ratio} < \mathbf{2.0\%}$.

---

### 3.4 Tower Shaft Core Dispersion ($\sigma_{\text{tower}}$)
- **Why It Is Needed**: Bridge towers are rigid vertical pylon structures. This metric verifies that tower points remain tightly clustered inside the structural $(u, w)$ cylinder along elevation axis $v$.
- **Target**: High core density with zero coplanar floating artifacts.

---

## 4. Multi-View Hold-Out Cross-Validation Protocol

To evaluate 3D semantic accuracy in the absence of full 3D ground truth:
1. **Hold-out Partition**: Partition the 300 labeled frames into **240 training views (80%)** and **60 hold-out test views (20%)**.
2. **3D Reconstruction**: Run Task A + Task B using only the 240 training views to reconstruct the colored 3D point cloud.
3. **Z-Buffer Re-projection**: Project classified 3D points onto the 60 hold-out camera poses:
   $$\mathbf{p}_{2D} = \pi(\mathbf{K}, \mathbf{R}_{\text{test}}, \mathbf{T}_{\text{test}}, \mathbf{X}_{3D})$$
4. **2D-3D Comparison**: Compare projected 2D masks against manual Labelme JSON ground-truth to compute unbiased $mIoU$.

---

## 5. Master Benchmark Summary Table

| Pillar | Metric | Mathematical Formulation | Recommended Benchmark Target | Primary Motivation & Why Needed |
| :--- | :--- | :--- | :---: | :--- |
| **Pillar 1: Semantics** | **Structural $mIoU$** | $\frac{1}{4}\sum_{c=1}^4 IoU_c$ | **$> 85.0\%$** | Penalizes class imbalance across all 4 bridge components |
| | **Cable $IoU$ ($IoU_{\text{cable}}$)** | $TP / (TP + FP + FN)$ | **$> 75.0\%$** | Measures resistance to background bleeding on slender cables |
| | **Overall Accuracy ($OA$)** | $\sum TP / N_{\text{total}}$ | **$> 92.0\%$** | Global point classification accuracy |
| **Pillar 2: Geometry** | **Mean Reprojection Error** | $\frac{1}{N}\sum \|p_{2D} - \pi(X_{3D})\|$ | **$< 1.0\text{ px}$** | Quantifies camera calibration and ray intersection accuracy |
| | **Point Cloud Completeness** | $\text{Count}(X_{\text{structural}})$ | **$> 75,000\text{ pts}$** | Ensures no structural holes in the bridge model |
| **Pillar 3: SHM Priors** | **Deck Planarity MAD** | $\text{median}(\|n \cdot x + d\|)$ | **$< 0.05\text{ m}$** | Evaluates roadway flatness and suppresses SfM depth drift |
| | **Cable Fan Deviation** | $\text{mean}(\text{dist}(x, \text{Fan Planes}))$ | **$< 0.10\text{ m}$** | Verifies cable alignment onto physical left/right fan sheets |
| | **Off-Fan Outlier Ratio** | $\text{Ratio}(\text{dist} > 0.10\text{m})$ | **$< 2.0\%$** | Quantifies elimination of floating sky/water ghost artifacts |
