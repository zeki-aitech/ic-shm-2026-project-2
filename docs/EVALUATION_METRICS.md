# IC-SHM 2026 (Project 2) — Evaluation Framework & Performance Metrics

Project 2 submissions are scored on a **blind test set of camera viewpoints** using exactly two
criteria (see `data/Contest Dataset/The 4th International Project Competition for
SHM_2026.pdf`, pp. 9-10):

| Criterion | What is compared | Metric(s) |
| :--- | :--- | :--- |
| **Visual Fidelity** | Rendered RGB image from a test viewpoint vs. the real photo | PSNR, SSIM, LPIPS |
| **Semantic Accuracy** | Rendered semantic map (official class IDs) vs. GT label map | mIoU |

$$\text{Accuracy Score} = 0.50 \times \text{Visual Fidelity Score} + 0.50 \times \text{Semantic mIoU Score}$$

## Implementation

`src/evaluation/render_metrics.py` implements this scoring:
- `compute_psnr` / `compute_ssim` use `skimage.metrics`.
- `compute_lpips` uses the `lpips` package.
- Semantic mIoU reuses `compute_confusion_matrix` / `compute_iou_per_class` / `compute_miou`
  from `src/evaluation/metrics.py`.
- `evaluate_render_holdout` / `RenderEvalReport` run all of the above on a set of held-out
  camera viewpoints and produce a Markdown report.

The exact intra-"Visual Fidelity" combination of PSNR/SSIM/LPIPS into one number is not
specified in the brief, so `RenderEvalReport` reports all three separately plus a clearly
labeled illustrative combination, alongside `Accuracy Score = 0.5 x Visual Fidelity + 0.5 x mIoU`.

The submitted deliverable that gets scored is `src/gaussian_splatting/render.py`: given any
camera pose, it renders an RGB PNG and a semantic-class PNG (official IDs 0-4) from the trained
`SemanticGaussianModel`.

## Semantic Classes

$mIoU_{\text{struct}}$ is averaged over the 4 structural classes, excluding background:

$$IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \qquad mIoU_{\text{struct}} = \frac{1}{4}\sum_{c=1}^{4} IoU_c$$

| Class ID | Label |
| :---: | :--- |
| 0 | background |
| 1 | deck |
| 2 | stay_cable |
| 3 | tower |
| 4 | foundation |

## Held-Out Evaluation Protocol

Evaluated on a **trajectory-interleaved 60-image holdout** split
(`src.evaluation.metrics.trajectory_interleaved_split`): the 300 labeled images are sorted along
the UAV flight trajectory, and every 5th image (60 total, 20%) is held out for evaluation — never
used in training the segmentation model or the Gaussian Splatting model, and never used even to
warm-start the Gaussian Splatting model's parameters.

This strided split spreads the held-out views evenly across the entire flight path rather than
sampling randomly, which would risk placing near-duplicate adjacent frames (>99% visual overlap)
on both sides of the split and inflating the measured score.
