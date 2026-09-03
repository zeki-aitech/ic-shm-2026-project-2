"""
Camera list construction for Gaussian Splatting training/eval, built directly from
`PycolmapReconstructor`'s output (`CameraIntrinsics`, `Dict[int, ImagePose]`) - one shared
pinhole `K` for all 400 views since the contest ships a single COLMAP camera.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from src.colmap_io.models import CameraIntrinsics, ImagePose
from src.gaussian_splatting.undistort import build_pinhole_K


@dataclass
class GSCamera:
    image_id: int
    name: str
    K: np.ndarray          # (3,3) pinhole intrinsics, post-undistortion
    R: np.ndarray           # (3,3) world -> camera rotation
    T: np.ndarray            # (3,)  world -> camera translation
    width: int
    height: int
    image_path: str
    mask_path: Optional[str]
    is_holdout: bool


def build_camera_list(
    camera: CameraIntrinsics,
    images: Dict[int, ImagePose],
    undistorted_dir: str,
    mask_lookup: Optional[Dict[str, str]] = None,
    holdout_stems: Optional[Set[str]] = None,
) -> List[GSCamera]:
    """
    `mask_lookup` maps an image stem (e.g. "001") to a mask PNG path (GT for labeled frames,
    pseudo-mask for unlabeled frames) - absent for frames with no mask available.
    `holdout_stems` marks the stems reserved for evaluation only (never trained on).
    """
    K = build_pinhole_K(camera)
    mask_lookup = mask_lookup or {}
    holdout_stems = holdout_stems or set()

    cams = []
    for image_id, pose in sorted(images.items()):
        stem = os.path.splitext(pose.name)[0]
        cams.append(
            GSCamera(
                image_id=image_id,
                name=pose.name,
                K=K,
                R=np.asarray(pose.R, dtype=np.float64).reshape(3, 3),
                T=np.asarray(pose.T, dtype=np.float64).reshape(3),
                width=camera.width,
                height=camera.height,
                image_path=os.path.join(undistorted_dir, pose.name),
                mask_path=mask_lookup.get(stem),
                is_holdout=stem in holdout_stems,
            )
        )
    return cams
