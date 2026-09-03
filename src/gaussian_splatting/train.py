"""
Task B: train the Semantic 3D Gaussian Splatting model.

Trains on the 240 trajectory-interleaved labeled images (GT masks) + the 100 unlabeled images
(pseudo-masks from Task A, see `src/segmentation/infer.py`) = up to 340 views. The 60 held-out
labeled images are never used here - reserved entirely for
`src/evaluation/render_metrics.py`'s final novel-view evaluation, mirroring the organizers'
blind-test protocol.

Gaussian means/colors are warm-started from `PycolmapReconstructor`'s triangulated sparse cloud,
and semantic logits from `SemanticProjector`'s per-point voted class (using ONLY the 240 train
views' masks - the 60 holdout never influences even the initialization).
"""
import argparse
import glob
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.colmap_io.reconstructor import PycolmapReconstructor
from src.colmap_io.semantic_voting import SemanticProjector
from src.evaluation.metrics import trajectory_interleaved_split
from src.gaussian_splatting.undistort import undistort_all
from src.gaussian_splatting.dataset import build_camera_list, GSCamera
from src.gaussian_splatting.model import SemanticGaussianModel, NUM_CLASSES
from src.gaussian_splatting.losses import photometric_loss, semantic_ce_loss

from gsplat.strategy import DefaultStrategy

PARAM_LRS = {
    "means": 1.6e-4,
    "scales": 5e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "colors": 2.5e-3,
    "sem_logits": 2.5e-3,
}


def _load_labeled_ids(images_dir: str) -> List[str]:
    ids = []
    for fname in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() == ".png" and stem.isdigit():
            ids.append(stem)
    return sorted(ids)


def _unlabeled_ids(unlabeled_dir: str) -> List[str]:
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(unlabeled_dir, "*.png"))
    )


def _camera_center(pose_R: np.ndarray, pose_T: np.ndarray) -> np.ndarray:
    return -pose_R.T @ pose_T


def _compute_scene_scale(cameras: List[GSCamera]) -> float:
    centers = np.stack([_camera_center(c.R, c.T) for c in cameras])
    extent = centers.max(axis=0) - centers.min(axis=0)
    return float(np.linalg.norm(extent) / 2.0 + 1e-6)


def _scale_camera(camera: GSCamera, scale: float) -> GSCamera:
    if scale == 1.0:
        return camera
    K = camera.K.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale
    return GSCamera(
        image_id=camera.image_id, name=camera.name, K=K, R=camera.R, T=camera.T,
        width=max(1, int(round(camera.width * scale))), height=max(1, int(round(camera.height * scale))),
        image_path=camera.image_path, mask_path=camera.mask_path, is_holdout=camera.is_holdout,
    )


def _load_rgb_tensor(path: str, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)


def _load_mask_tensor(path: str, size: Tuple[int, int]) -> torch.Tensor:
    m = Image.open(path).resize(size, Image.NEAREST)
    return torch.from_numpy(np.asarray(m, dtype=np.int64))


class _CachedParser:
    """Adapts an already-loaded (camera, images, pts3d) tuple to `SemanticProjector`'s
    `parser.load()` contract, so it doesn't silently re-run LO-RANSAC triangulation a second
    time when we've already loaded the reconstruction once in `prepare_training_data`."""

    def __init__(self, camera, images, pts3d):
        self._data = (camera, images, pts3d)

    def load(self):
        return self._data


def prepare_training_data(
    colmap_dir: str,
    images_dir: str,
    unlabeled_dir: str,
    gt_masks_dir: str,
    pseudo_masks_dir: Optional[str],
    undistorted_dir: str,
    holdout_ratio: float = 0.2,
):
    """Loads camera/points/votes, builds the train (labeled+unlabeled) and holdout camera lists.
    Returns (camera_intrinsics, pts3d, point_classes, point_colors, train_cameras, holdout_cameras,
    train_ids)."""
    camera, images, pts3d = PycolmapReconstructor(colmap_dir).load()

    labeled_ids = _load_labeled_ids(images_dir)
    train_ids, holdout_ids = trajectory_interleaved_split(labeled_ids, holdout_ratio)
    holdout_set = set(holdout_ids)
    print(f"[gaussian_splatting] labeled={len(labeled_ids)} train={len(train_ids)} holdout={len(holdout_ids)}")

    if not os.path.isdir(undistorted_dir) or not os.listdir(undistorted_dir):
        print(f"[gaussian_splatting] undistorting images -> {undistorted_dir}")
        undistort_all([images_dir, unlabeled_dir], camera, undistorted_dir)

    # Vote per-3D-point semantic classes using ONLY the train-split masks (holdout stays
    # completely untouched, even for this warm-start step): pre-populate the mask cache with
    # just the train-split GT masks, keyed by path exactly as `SemanticProjector` expects, so
    # its internal `preload_masks()` fallback (which would load every mask in the directory)
    # never fires.
    projector = SemanticProjector(
        colmap_dir, gt_masks_dir,
        parser=_CachedParser(camera, images, pts3d),
    )
    for stem in train_ids:
        mask_path = os.path.join(gt_masks_dir, f"{stem}.png")
        if os.path.exists(mask_path):
            projector.mask_cache[mask_path] = np.array(Image.open(mask_path), dtype=np.uint8)
    point_classes, point_colors = projector.project()

    # `mask_lookup` covers train (GT), holdout (GT - eval-only, never used by `train_cameras`
    # since those are additionally filtered on `not is_holdout`) and pseudo-labeled unlabeled ids.
    mask_lookup: Dict[str, str] = {
        stem: os.path.join(gt_masks_dir, f"{stem}.png") for stem in list(train_ids) + list(holdout_ids)
    }
    if pseudo_masks_dir and os.path.isdir(pseudo_masks_dir):
        for stem in _unlabeled_ids(unlabeled_dir):
            pseudo_path = os.path.join(pseudo_masks_dir, f"{stem}.png")
            if os.path.exists(pseudo_path):
                mask_lookup[stem] = pseudo_path

    all_cameras = build_camera_list(camera, images, undistorted_dir, mask_lookup, holdout_set)
    train_cameras = [c for c in all_cameras if not c.is_holdout and c.mask_path is not None]
    holdout_cameras = [c for c in all_cameras if c.is_holdout]

    return camera, pts3d, point_classes, point_colors, train_cameras, holdout_cameras, set(train_ids)


def train(
    colmap_dir: str,
    images_dir: str,
    unlabeled_dir: str,
    gt_masks_dir: str,
    pseudo_masks_dir: Optional[str],
    undistorted_dir: str,
    output_dir: str,
    holdout_ratio: float = 0.2,
    iters: int = 20000,
    downsample: float = 0.5,
    lambda_sem: float = 0.5,
    pseudo_mask_weight: float = 0.5,
    max_gaussians: int = 600_000,
    log_every: int = 100,
    ckpt_every: int = 2000,
    device: str = None,
    seed: int = 42,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    camera_intr, pts3d, point_classes, point_colors, train_cameras, holdout_cameras, train_ids = prepare_training_data(
        colmap_dir, images_dir, unlabeled_dir, gt_masks_dir, pseudo_masks_dir, undistorted_dir, holdout_ratio
    )
    print(f"[gaussian_splatting] train views={len(train_cameras)} holdout views={len(holdout_cameras)}")

    model = SemanticGaussianModel.init_from_sparse(pts3d, point_classes, point_colors, device=device)
    print(f"[gaussian_splatting] initial Gaussians: {model.num_points}")

    optimizers = {k: torch.optim.Adam([v], lr=PARAM_LRS[k], eps=1e-15) for k, v in model.params.items()}
    means_lr_init, means_lr_final = PARAM_LRS["means"], PARAM_LRS["means"] * 0.01

    scene_scale = _compute_scene_scale(train_cameras)
    strategy = DefaultStrategy(
        refine_start_iter=500,
        refine_stop_iter=min(15000, int(iters * 0.75)),
        refine_every=100,
        reset_every=3000,
        verbose=False,
    )
    strategy.check_sanity(model.params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    # Preload (downsampled) train images/masks into memory once - avoids repeated disk I/O.
    cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, GSCamera, float]] = {}
    for cam in train_cameras:
        scaled_cam = _scale_camera(cam, downsample)
        size = (scaled_cam.width, scaled_cam.height)
        rgb = _load_rgb_tensor(cam.image_path, size)
        mask = _load_mask_tensor(cam.mask_path, size)
        weight = 1.0 if cam.name.split(".")[0] in train_ids else pseudo_mask_weight
        cache[cam.name] = (rgb, mask, scaled_cam, weight)

    order = list(cache.keys())
    t0 = time.time()
    for step in range(1, iters + 1):
        if step % len(order) == 1:
            random.shuffle(order)
        name = order[(step - 1) % len(order)]
        rgb_target, mask_target, cam, sem_weight = cache[name]
        rgb_target = rgb_target.to(device)
        mask_target = mask_target.to(device)

        for g in optimizers["means"].param_groups:
            frac = min(1.0, step / max(1, iters))
            g["lr"] = means_lr_init * (means_lr_final / means_lr_init) ** frac

        out, _alpha, meta = model.render_full(cam, packed=True)
        strategy.step_pre_backward(model.params, optimizers, state, step, meta)

        out0 = out[0]
        rgb_pred = out0[..., :3].clamp(0.0, 1.0)
        sem_pred = out0[..., 3:]

        loss = photometric_loss(rgb_pred, rgb_target) + lambda_sem * sem_weight * semantic_ce_loss(sem_pred, mask_target)
        loss.backward()

        if model.num_points < max_gaussians:
            strategy.step_post_backward(model.params, optimizers, state, step, meta, packed=True)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % log_every == 0:
            elapsed = time.time() - t0
            print(f"[gaussian_splatting] step {step}/{iters} loss={loss.item():.4f} "
                  f"n_gaussians={model.num_points} ({elapsed:.1f}s)")

        if step % ckpt_every == 0 or step == iters:
            ckpt_path = os.path.join(output_dir, f"step_{step}.pt")
            torch.save({"params": model.state_dict(), "step": step}, ckpt_path)

    final_path = os.path.join(output_dir, "final.pt")
    torch.save({"params": model.state_dict(), "step": iters}, final_path)
    print(f"[gaussian_splatting] training complete -> {final_path}")
    return model, final_path, holdout_cameras


def main():
    parser = argparse.ArgumentParser(description="Task B: train the Semantic 3D Gaussian Splatting model")
    parser.add_argument("--colmap-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--unlabeled-dir", default=None)
    parser.add_argument("--gt-masks-dir", default=None)
    parser.add_argument("--pseudo-masks-dir", default=None)
    parser.add_argument("--undistorted-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--downsample", type=float, default=0.5)
    parser.add_argument("--lambda-sem", type=float, default=0.5)
    args = parser.parse_args()

    dataset_dir = os.getenv("CONTEST_DATASET_DIR", os.path.join(PROJECT_ROOT, "data", "Contest Dataset"))
    colmap_dir = args.colmap_dir or os.path.join(dataset_dir, "camera_parameters")
    images_dir = args.images_dir or os.path.join(dataset_dir, "images")
    unlabeled_dir = args.unlabeled_dir or os.path.join(dataset_dir, "unlabeled_Images")
    gt_masks_dir = args.gt_masks_dir or os.path.join(PROJECT_ROOT, "outputs", "gt_masks")
    pseudo_masks_dir = args.pseudo_masks_dir or os.path.join(PROJECT_ROOT, "outputs", "pseudo_masks")
    undistorted_dir = args.undistorted_dir or os.path.join(PROJECT_ROOT, "outputs", "undistorted_images")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "gaussians")

    train(
        colmap_dir, images_dir, unlabeled_dir, gt_masks_dir, pseudo_masks_dir, undistorted_dir, output_dir,
        holdout_ratio=args.holdout_ratio, iters=args.iters, downsample=args.downsample, lambda_sem=args.lambda_sem,
    )


if __name__ == "__main__":
    main()
