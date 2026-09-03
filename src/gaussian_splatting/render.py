"""
Task B: the literal contest submission deliverable. Renders an RGB image and a semantic map
(official class IDs 0-4) from an arbitrary camera viewpoint using a trained Semantic Gaussian
Splatting checkpoint, so organizers can automatically evaluate against their blind held-out test
poses (see `data/Contest Dataset/The 4th International Project Competition for SHM_2026.pdf`,
p.10, "Submission Requirements").
"""
import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.colmap_io.models import CameraIntrinsics
from src.gaussian_splatting.model import SemanticGaussianModel
from src.gaussian_splatting.dataset import GSCamera
from src.gaussian_splatting.undistort import build_pinhole_K


def load_trained_model(checkpoint_path: str, device: str = "cuda") -> SemanticGaussianModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return SemanticGaussianModel.from_state_dict(ckpt["params"], device=device)


def load_camera_intrinsics(colmap_dir: str) -> CameraIntrinsics:
    """Reads just the shared camera intrinsics from `cameras.txt` without triangulating the
    sparse point cloud (which `PycolmapReconstructor.load()` would do, at real but unnecessary
    cost for a render-only call)."""
    from src.colmap_io.reconstructor import load_contest_model, reconstruction_to_parser_data

    rec = load_contest_model(colmap_dir)
    camera, _images, _pts3d = reconstruction_to_parser_data(rec)
    return camera


def quat_translation_to_Rt(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float):
    """COLMAP convention: (qw,qx,qy,qz)/(tx,ty,tz) is the CAMERA_FROM_WORLD pose - exactly the
    `R`/`T` fields of `ImagePose`, and what `GSCamera.R`/`GSCamera.T` expect."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    T = np.array([tx, ty, tz], dtype=np.float64)
    return R, T


def parse_images_txt_pose_line(line: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parses one COLMAP `images.txt` pose line: `IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`."""
    parts = line.strip().split()
    qw, qx, qy, qz, tx, ty, tz = (float(v) for v in parts[1:8])
    return quat_translation_to_Rt(qw, qx, qy, qz, tx, ty, tz)


def render_view(
    model: SemanticGaussianModel,
    camera_intrinsics: CameraIntrinsics,
    R: np.ndarray,
    T: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Renders `(rgb uint8 [H,W,3], semantic_class_id uint8 [H,W])` for an arbitrary camera
    pose, using the official class IDs (0=background,1=deck,2=stay_cable,3=tower,4=foundation)."""
    K = build_pinhole_K(camera_intrinsics)
    camera = GSCamera(
        image_id=-1, name="render", K=K, R=R, T=T,
        width=width or camera_intrinsics.width, height=height or camera_intrinsics.height,
        image_path="", mask_path=None, is_holdout=False,
    )
    with torch.no_grad():
        rgb, sem_logits = model.render(camera)
    rgb_np = (rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    sem_np = sem_logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
    return rgb_np, sem_np


def main():
    parser = argparse.ArgumentParser(
        description="Task B: render RGB + semantic map from an arbitrary camera viewpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--colmap-dir", default=None, help="Only used to read the shared camera intrinsics")
    parser.add_argument("--pose-line", default=None, help="A COLMAP images.txt pose line: 'ID QW QX QY QZ TX TY TZ CAM_ID NAME'")
    parser.add_argument("--qvec", nargs=4, type=float, default=None, metavar=("QW", "QX", "QY", "QZ"))
    parser.add_argument("--tvec", nargs=3, type=float, default=None, metavar=("TX", "TY", "TZ"))
    parser.add_argument("--out-rgb", required=True)
    parser.add_argument("--out-sem", required=True)
    args = parser.parse_args()

    dataset_dir = os.getenv("CONTEST_DATASET_DIR", os.path.join(PROJECT_ROOT, "data", "Contest Dataset"))
    colmap_dir = args.colmap_dir or os.path.join(dataset_dir, "camera_parameters")
    camera_intrinsics = load_camera_intrinsics(colmap_dir)

    if args.pose_line:
        R, T = parse_images_txt_pose_line(args.pose_line)
    elif args.qvec and args.tvec:
        R, T = quat_translation_to_Rt(*args.qvec, *args.tvec)
    else:
        raise ValueError("Provide either --pose-line or both --qvec and --tvec")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(args.checkpoint, device=device)
    rgb, sem = render_view(model, camera_intrinsics, R, T)

    for out_path in (args.out_rgb, args.out_sem):
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(args.out_rgb)
    Image.fromarray(sem, mode="L").save(args.out_sem)
    print(f"[render] wrote {args.out_rgb} and {args.out_sem}")


if __name__ == "__main__":
    main()
