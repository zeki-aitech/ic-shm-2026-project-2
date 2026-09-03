from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for perspective projection."""
    id: int
    model: str
    width: int
    height: int
    f: float
    cx: float
    cy: float
    k1: float = 0.0


@dataclass
class ImagePose:
    """6-DOF camera extrinsic pose and 2D observation tracks."""
    image_id: int
    name: str
    qvec: np.ndarray      # [qw, qx, qy, qz]
    tvec: np.ndarray      # [tx, ty, tz]
    R: np.ndarray         # 3x3 rotation matrix (Camera from World)
    T: np.ndarray         # 3x1 translation vector (Camera from World)
    P: np.ndarray         # 3x4 projection matrix [R | T]
    camera_id: int
    points2d: List[Tuple[float, float, int]] = field(default_factory=list)  # list of (x, y, point3d_id)


@dataclass
class Point3D:
    """Reconstructed 3D world coordinate and associated 2D track observations."""
    id: int
    xyz: np.ndarray       # 3D coordinate (x, y, z) in world space
    image_ids: List[int]  # List of observing image IDs
    point2d_idxs: List[int] # List of 2D feature indices in respective images
