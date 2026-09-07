"""
Renders a smooth camera path interpolated between two poses that were actually flown by the UAV,
for Figure 5 (Section 5.3) - demonstrating rendering from a genuinely arbitrary viewpoint (not
merely a pose selected from the acquisition trajectory), with RGB and semantic outputs staying
pixel-aligned as the camera moves.

Rotation is interpolated via SLERP on the unit quaternion, translation via linear interpolation,
each independently parameterized by the same t in [0, 1] - a standard, simple choice for a short
illustrative path (not a claim of geodesic-optimal camera motion).
"""
import os
import sys
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.colmap_io.models import CameraIntrinsics
from src.gaussian_splatting.model import SemanticGaussianModel
from src.gaussian_splatting.render import quat_translation_to_Rt, render_view


def slerp_quat(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions (w,x,y,z), taking the
    shorter arc."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q_perp = q2 - q1 * dot
    q_perp = q_perp / np.linalg.norm(q_perp)
    return q1 * np.cos(theta) + q_perp * np.sin(theta)


def interpolate_pose(
    qvec1: Tuple[float, float, float, float], tvec1: Tuple[float, float, float],
    qvec2: Tuple[float, float, float, float], tvec2: Tuple[float, float, float],
    t: float,
):
    """Returns (R, T) for the pose at interpolation parameter `t` in [0, 1] between the two
    given COLMAP-convention (qw,qx,qy,qz)/(tx,ty,tz) poses."""
    q1, q2 = np.asarray(qvec1, dtype=np.float64), np.asarray(qvec2, dtype=np.float64)
    tv1, tv2 = np.asarray(tvec1, dtype=np.float64), np.asarray(tvec2, dtype=np.float64)
    q = slerp_quat(q1, q2, t)
    tv = (1.0 - t) * tv1 + t * tv2
    return quat_translation_to_Rt(*q, *tv)


def render_interpolated_path(
    model: SemanticGaussianModel,
    camera_intrinsics: CameraIntrinsics,
    qvec1, tvec1, qvec2, tvec2,
    n_steps: int = 5,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Renders `n_steps` evenly-spaced frames (including both endpoints) along the interpolated
    path. Returns a list of `(t, rgb, sem)`."""
    frames = []
    for i in range(n_steps):
        t = i / (n_steps - 1) if n_steps > 1 else 0.0
        R, T = interpolate_pose(qvec1, tvec1, qvec2, tvec2, t)
        rgb, sem = render_view(model, camera_intrinsics, R, T)
        frames.append((t, rgb, sem))
    return frames


def main():
    import argparse

    from PIL import Image

    from src.gaussian_splatting.render import load_camera_intrinsics, load_trained_model

    parser = argparse.ArgumentParser(
        description="Render a camera path interpolated between two real flown poses (Figure 5)"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--colmap-dir", default=None)
    parser.add_argument("--pose-line-a", required=True, help="COLMAP images.txt pose line for endpoint A")
    parser.add_argument("--pose-line-b", required=True, help="COLMAP images.txt pose line for endpoint B")
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import torch

    dataset_dir = os.getenv("CONTEST_DATASET_DIR", os.path.join(PROJECT_ROOT, "data", "Contest Dataset"))
    colmap_dir = args.colmap_dir or os.path.join(dataset_dir, "camera_parameters")
    camera_intrinsics = load_camera_intrinsics(colmap_dir)

    def _parse(line):
        parts = line.strip().split()
        qw, qx, qy, qz, tx, ty, tz = (float(v) for v in parts[1:8])
        return (qw, qx, qy, qz), (tx, ty, tz)

    qvec1, tvec1 = _parse(args.pose_line_a)
    qvec2, tvec2 = _parse(args.pose_line_b)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(args.checkpoint, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    frames = render_interpolated_path(model, camera_intrinsics, qvec1, tvec1, qvec2, tvec2, args.n_steps)
    for i, (t, rgb, sem) in enumerate(frames):
        Image.fromarray(rgb, mode="RGB").save(os.path.join(args.output_dir, f"t{i:02d}_{t:.2f}_rgb.png"))
        Image.fromarray(sem, mode="L").save(os.path.join(args.output_dir, f"t{i:02d}_{t:.2f}_sem.png"))
    print(f"[interpolate] wrote {args.n_steps} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
