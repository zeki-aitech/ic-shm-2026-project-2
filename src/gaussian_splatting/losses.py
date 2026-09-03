"""Photometric (L1 + D-SSIM) and semantic cross-entropy losses for Gaussian Splatting training."""
import torch
import torch.nn.functional as F


def l1_loss(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    """`pred_rgb`/`target_rgb`: (H, W, 3) in [0, 1]."""
    return torch.abs(pred_rgb - target_rgb).mean()


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.outer(g)
    return window_2d.unsqueeze(0).unsqueeze(0)  # (1,1,K,K)


def d_ssim_loss(pred_rgb: torch.Tensor, target_rgb: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """`1 - SSIM`, i.e. the D-SSIM term used in the standard 3DGS photometric loss.
    `pred_rgb`/`target_rgb`: (H, W, 3) in [0, 1]."""
    pred = pred_rgb.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    target = target_rgb.permute(2, 0, 1).unsqueeze(0)

    window = _gaussian_window(window_size, sigma=1.5, device=pred.device, dtype=pred.dtype)
    window = window.expand(3, 1, window_size, window_size)
    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=3)
    mu_target = F.conv2d(target, window, padding=pad, groups=3)
    mu_pred_sq, mu_target_sq, mu_pred_target = mu_pred**2, mu_target**2, mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=3) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=3) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=3) - mu_pred_target

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return 1.0 - ssim_map.mean()


def photometric_loss(
    pred_rgb: torch.Tensor, target_rgb: torch.Tensor, lambda_ssim: float = 0.2
) -> torch.Tensor:
    return (1.0 - lambda_ssim) * l1_loss(pred_rgb, target_rgb) + lambda_ssim * d_ssim_loss(pred_rgb, target_rgb)


def semantic_ce_loss(pred_logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
    """`pred_logits`: (H, W, num_classes). `target_labels`: (H, W) long class ids."""
    logits_flat = pred_logits.reshape(-1, pred_logits.shape[-1])
    labels_flat = target_labels.reshape(-1).long()
    return F.cross_entropy(logits_flat, labels_flat)
