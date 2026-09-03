"""
Render-based evaluation matching the official contest scoring protocol (see
`data/Contest Dataset/The 4th International Project Competition for SHM_2026.pdf`, pp. 9-10):
for held-out camera viewpoints, render RGB + a semantic map and compare against ground truth via
PSNR / SSIM / LPIPS ("Visual Fidelity") and mIoU ("Semantic Accuracy").

`compute_psnr`/`compute_ssim` only need numpy + scikit-image (already project dependencies).
`compute_lpips` and `evaluate_render_holdout` lazily import torch/lpips/the Gaussian model so
this module stays importable without the `deeplearning` extra for pure visual-metric use.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.evaluation.metrics import (
    CLASS_NAMES,
    compute_confusion_matrix,
    compute_iou_per_class,
    compute_miou,
)

_LPIPS_MODEL = None


def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def compute_psnr(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    """`pred_rgb`/`gt_rgb`: (H,W,3), uint8 [0,255] or float [0,1]."""
    pred, gt = _to_float01(pred_rgb), _to_float01(gt_rgb)
    return float(peak_signal_noise_ratio(gt, pred, data_range=1.0))


def compute_ssim(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    pred, gt = _to_float01(pred_rgb), _to_float01(gt_rgb)
    return float(structural_similarity(gt, pred, data_range=1.0, channel_axis=-1))


def _get_lpips_model(net: str = "alex", device: str = "cpu"):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips

        _LPIPS_MODEL = lpips.LPIPS(net=net).to(device)
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL


def compute_lpips(pred_rgb: np.ndarray, gt_rgb: np.ndarray, net: str = "alex", device: str = "cpu") -> float:
    import torch

    model = _get_lpips_model(net, device)
    pred, gt = _to_float01(pred_rgb), _to_float01(gt_rgb)
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        dist = model(pred_t.to(device), gt_t.to(device))
    return float(dist.item())


@dataclass
class RenderEvalReport:
    psnr: float
    ssim: float
    lpips: float
    miou: float
    iou_per_class: Dict[int, float]
    n_views: int
    visual_fidelity_illustrative: float
    accuracy_score_illustrative: float
    per_view: List[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Render-Based Evaluation Report (Official Contest Protocol)",
            "",
            f"Evaluated on {self.n_views} held-out camera viewpoints (never used in training).",
            "",
            "## Visual Fidelity",
            f"- PSNR: {self.psnr:.3f} dB",
            f"- SSIM: {self.ssim:.4f}",
            f"- LPIPS: {self.lpips:.4f} (lower is better)",
            "",
            "## Semantic Accuracy",
            f"- mIoU (structural, 4 classes, background excluded): {self.miou:.4f}",
        ]
        for cid, iou in sorted(self.iou_per_class.items()):
            lines.append(f"  - {CLASS_NAMES.get(cid, cid)}: {iou:.4f}")
        lines += [
            "",
            "## Illustrative Accuracy Score",
            "*The contest brief defines `Accuracy Score = 0.5 x Visual Fidelity + 0.5 x Semantic mIoU`",
            "but does not specify how PSNR/SSIM/LPIPS combine into one Visual Fidelity number - this",
            "combination is our own documented choice for internal tracking, not an authoritative",
            "organizer formula.*",
            f"- Visual Fidelity (illustrative): {self.visual_fidelity_illustrative:.4f}",
            f"- Accuracy Score (illustrative): {self.accuracy_score_illustrative:.4f}",
        ]
        return "\n".join(lines)


def _illustrative_visual_fidelity(psnr: float, ssim: float, lpips_val: float) -> float:
    """
    Documented, non-authoritative combination of the three visual-fidelity metrics into [0,1]:
    PSNR normalized against a 35dB reference (typical "good" novel-view PSNR for this kind of
    scene), SSIM used directly (already in [0,1]), LPIPS inverted (lower is better -> 1-LPIPS).
    """
    psnr_norm = float(np.clip(psnr / 35.0, 0.0, 1.0))
    lpips_norm = float(np.clip(1.0 - lpips_val, 0.0, 1.0))
    return float(np.mean([psnr_norm, ssim, lpips_norm]))


def evaluate_render_holdout(
    model,
    holdout_cameras: List,
    device: str = "cuda",
    lpips_net: str = "alex",
) -> RenderEvalReport:
    """
    `model`: a trained `SemanticGaussianModel`.
    `holdout_cameras`: `GSCamera` list for the held-out views (each must have `mask_path` set to
    its GT mask and `image_path` pointing at the real undistorted photo).
    """
    import torch
    from PIL import Image

    conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    psnrs, ssims, lpipss = [], [], []
    per_view = []

    for camera in holdout_cameras:
        with torch.no_grad():
            rgb, sem_logits = model.render(camera)
        pred_rgb = (rgb.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pred_mask = sem_logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)

        gt_rgb = np.asarray(Image.open(camera.image_path).convert("RGB"))
        gt_mask = np.asarray(Image.open(camera.mask_path))

        p = compute_psnr(pred_rgb, gt_rgb)
        s = compute_ssim(pred_rgb, gt_rgb)
        l = compute_lpips(pred_rgb, gt_rgb, net=lpips_net, device=device)
        psnrs.append(p)
        ssims.append(s)
        lpipss.append(l)

        conf += compute_confusion_matrix(gt_mask.ravel(), pred_mask.ravel(), len(CLASS_NAMES))
        per_view.append({"name": camera.name, "psnr": p, "ssim": s, "lpips": l})

    ious = compute_iou_per_class(conf)
    miou = compute_miou(ious, include_background=False)

    mean_psnr, mean_ssim, mean_lpips = float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipss))
    visual_fidelity = _illustrative_visual_fidelity(mean_psnr, mean_ssim, mean_lpips)
    accuracy_score = 0.5 * visual_fidelity + 0.5 * miou

    return RenderEvalReport(
        psnr=mean_psnr,
        ssim=mean_ssim,
        lpips=mean_lpips,
        miou=miou,
        iou_per_class=ious,
        n_views=len(holdout_cameras),
        visual_fidelity_illustrative=visual_fidelity,
        accuracy_score_illustrative=accuracy_score,
        per_view=per_view,
    )


def main():
    import argparse
    import os

    import torch

    from src.gaussian_splatting.model import SemanticGaussianModel
    from src.gaussian_splatting.train import prepare_training_data

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(
        description="Render-based evaluation on the 60 held-out (never-trained-on) views"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--colmap-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--unlabeled-dir", default=None)
    parser.add_argument("--gt-masks-dir", default=None)
    parser.add_argument("--undistorted-dir", default=None)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_dir = os.getenv("CONTEST_DATASET_DIR", os.path.join(project_root, "data", "Contest Dataset"))
    colmap_dir = args.colmap_dir or os.path.join(dataset_dir, "camera_parameters")
    images_dir = args.images_dir or os.path.join(dataset_dir, "images")
    unlabeled_dir = args.unlabeled_dir or os.path.join(dataset_dir, "unlabeled_Images")
    gt_masks_dir = args.gt_masks_dir or os.path.join(project_root, "outputs", "gt_masks")
    undistorted_dir = args.undistorted_dir or os.path.join(project_root, "outputs", "undistorted_images")
    output_path = args.output or os.path.join(project_root, "outputs", "eval", "render_eval_report.md")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, _, _, _, holdout_cameras, _ = prepare_training_data(
        colmap_dir, images_dir, unlabeled_dir, gt_masks_dir, None, undistorted_dir, args.holdout_ratio
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SemanticGaussianModel.from_state_dict(ckpt["params"], device=device)

    report = evaluate_render_holdout(model, holdout_cameras, device=device)
    print(report.to_markdown())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())
    print(f"\n[render_metrics] wrote {output_path}")


if __name__ == "__main__":
    main()
