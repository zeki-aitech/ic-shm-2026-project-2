# IC-SHM 2026 (Project 2) — Evaluation Framework & Performance Metrics

This document outlines the comprehensive evaluation methodology for the **Structure-Aware 3D Semantic Point Cloud Reconstruction of Cable-Stayed Bridges**. The evaluation protocol integrates **3D Semantic Segmentation Metrics**, **3D Geometric Reconstruction Accuracy**, and **Domain-Specific Structural Health Monitoring (SHM) Constraints**.

---

## 1. 3D Semantic Segmentation Metrics

Evaluates how accurately each 3D point is categorized into its corresponding structural class:
`0: background`, `1: deck`, `2: stay_cable`, `3: tower`, `4: foundation`.

### 1.1 Mean Intersection over Union (mIoU) — *Primary Metric*
The Intersection over Union ($IoU$) for each class $c \in \{0, 1, 2, 3, 4\}$ is defined as:
$$IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}$$
where $TP_c, FP_c, FN_c$ denote true positives, false positives, and false negatives for class $c$, respectively.

The Mean IoU ($mIoU$) across all $C$ classes is:
$$mIoU = \frac{1}{C} \sum_{c=1}^{C} IoU_c$$

> **Note on Class Imbalance**: In cable-stayed bridges, `deck` points outnumber `stay_cable` points by an order of magnitude. $IoU_{\text{cable}}$ is the most critical benchmark for assessing model fidelity on slender components.

---

### 1.2 Overall Accuracy (OA) & Mean Class Accuracy (mAcc)
- **Overall Accuracy (OA)**:
  $$OA = \frac{\sum_{c=1}^C TP_c}{N_{\text{total}}}$$
- **Mean Class Accuracy (mAcc / Mean Recall)**:
  $$mAcc = \frac{1}{C} \sum_{c=1}^C \frac{TP_c}{TP_c + FN_c}$$

---

### 1.3 Per-Class Precision, Recall, and $F_1$-Score
$$Precision_c = \frac{TP_c}{TP_c + FP_c}, \quad Recall_c = \frac{TP_c}{TP_c + FN_c}, \quad F_1^{(c)} = \frac{2 \cdot Precision_c \cdot Recall_c}{Precision_c + Recall_c}$$

---

## 2. 3D Geometric Reconstruction Accuracy

Evaluates the spatial fidelity, geometric precision, and noise suppression of the reconstructed 3D point cloud coordinates $(x, y, z)$.

### 2.1 Mean Reprojection Error
Measures how accurately 3D points project back onto the 2D image planes of calibrated cameras:
$$e_{\text{reproj}} = \frac{1}{N} \sum_{i=1}^N \| p_{2D}^{(i)} - \pi(K, R_k, T_k, X_{3D}^{(i)}) \|_2$$
where $\pi(\cdot)$ denotes the perspective camera projection function. A robust reconstruction achieves $e_{\text{reproj}} < 1.0 \text{ px}$.

---

### 2.2 Chamfer Distance (CD)
When LiDAR or as-designed CAD/BIM reference point clouds ($G$) are available against predicted points ($P$):
$$d_{CD}(P, G) = \frac{1}{|P|} \sum_{x \in P} \min_{y \in G} \|x - y\|_2^2 + \frac{1}{|G|} \sum_{y \in G} \min_{x \in P} \|x - y\|_2^2$$

---

### 2.3 Precision, Completeness, and $F$-Score at Distance Threshold $d_{th}$ (e.g., $d_{th} = 0.05\text{ m}$)
- **Geometric Precision (Accuracy)**: Percentage of predicted points within distance $d_{th}$ from the ground truth surface.
- **Geometric Recall (Completeness)**: Percentage of the ground truth surface covered by predicted points within $d_{th}$.
- **F-score**:
  $$F(d_{th}) = \frac{2 \cdot \text{Precision}(d_{th}) \cdot \text{Completeness}(d_{th})}{\text{Precision}(d_{th}) + \text{Completeness}(d_{th})}$$

---

## 3. Structure-Aware & Domain-Specific Metrics

Assesses adherence to physical, mechanical, and architectural properties of cable-stayed bridges.

| Component | Target Property | Metric & Mathematical Formulation | Ideal Target |
| :--- | :--- | :--- | :--- |
| **Bridge Deck (`deck`)** | **Planarity & Surface Smoothness** | 2-pass PCA plane residual MAD (Median Absolute Deviation): $\text{MAD} = \text{median}(\|n \cdot x + d\|)$ | $< 0.05\text{ m}$ |
| **Stay Cables (`stay_cable`)** | **Fan Planarity & Alignment** | Mean distance to the two tower-anchored fan planes: $\frac{1}{|P_{\text{cable}}|}\sum_{x \in P_{\text{cable}}} \min(|(x - x_0) \cdot w - d_{\text{left}}|, |(x - x_0) \cdot w - d_{\text{right}}|)$ | $< 0.10\text{ m}$ |
| **Stay Cables (`stay_cable`)** | **Off-Fan Outlier Ratio** | Proportion of cable points lying outside tolerance $\tau$: $\frac{1}{|P_{\text{cable}}|}\sum [dist > \tau]$ | $< 2.0\%$ |
| **Tower Pylons (`tower`)** | **Shaft Verticality & Compactness** | Radial dispersion $\sigma_{\text{radial}}$ around the longitudinal-lateral centroid tube of each tower shaft | High density core |
| **Global Structure** | **Topological Bounding Validity** | Percentage of cable points satisfying elevation constraint: $h_{\text{deck}} \le z_{\text{cable}} \le h_{\text{tower\_apex}}$ | $100\%$ valid |

---

## 4. Evaluation Protocol with 2D Ground-Truth (Hold-Out View Cross-Validation)

In the absence of dense 3D LiDAR scans, the multi-view reconstruction is validated using 2D ground-truth masks:

```text
 ┌───────────────────────────┐      ┌───────────────────────────┐
 │ Training Views (e.g. 80%) │      │  Hold-Out Views (e.g. 20%)│
 └─────────────┬─────────────┘      └─────────────┬─────────────┘
               │                                  │
               ▼                                  │
 ┌───────────────────────────┐                    │
 │ 3D Reconstruction &       │                    │
 │ Semantic Projection       │                    │
 └─────────────┬─────────────┘                    │
               │                                  │
               ▼                                  ▼
 ┌───────────────────────────┐      ┌───────────────────────────┐
 │ Re-project Colored 3D     │ ───► │ Compute 2D Mask IoU &     │
 │ Points to Hold-Out Frames │      │ Multi-View Consistency    │
 └───────────────────────────┘      └───────────────────────────┘
```

1. **2D Re-projection Cross-Validation**:
   - Re-project the classified 3D point cloud onto hold-out camera poses using z-buffering.
   - Compute **2D mIoU** and **Pixel Accuracy** against the manual Labelme ground-truth masks.
2. **Multi-View Consensus Entropy**:
   - For every 3D track observed across $K \ge 3$ images, compute the voting entropy $H(p) = -\sum_{c} p_c \log p_c$.
   - Lower entropy indicates consistent agreement across independent UAV viewpoints.

---

## 5. Summary Evaluation Scorecard

| Category | Metric | Recommended Target |
| :--- | :--- | :--- |
| **Semantic Classification** | Mean IoU ($mIoU$) | $> 85.0\%$ |
| | Stay Cable IoU ($IoU_{\text{cable}}$) | $> 75.0\%$ |
| | Overall Accuracy ($OA$) | $> 92.0\%$ |
| **Geometric Precision** | Mean Reprojection Error | $< 1.0\text{ px}$ |
| | Total Clean Structural Points | $> 75,000\text{ points}$ |
| **Structural Fidelity** | Deck Plane Residual MAD | $< 0.05\text{ m}$ |
| | Cable Fan Plane Deviation | $< 0.10\text{ m}$ |
| | Cable Off-Fan Outlier Rate | $< 2.0\%$ |
