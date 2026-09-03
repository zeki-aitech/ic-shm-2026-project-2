"""
One-time lens-undistortion pass over the UAV images.

The shared camera is COLMAP `SIMPLE_RADIAL` (f, cx, cy, k1 != 0). `gsplat`'s rasterizer models
an ideal pinhole camera and does not undistort internally, so training/eval/rendering must all
operate in a consistent undistorted-pinhole convention - this module produces it once and caches
the result, rather than leaving distortion handling implicit (small k1, but non-zero: skipping
this would silently corrupt multi-view geometry, most visibly near image edges).
"""
import os
from typing import Iterable, List

import cv2
import numpy as np

from src.colmap_io.models import CameraIntrinsics


def build_distorted_K(camera: CameraIntrinsics) -> np.ndarray:
    """The camera's own (still-distorted) intrinsic matrix, as shipped in cameras.txt."""
    return np.array(
        [[camera.f, 0.0, camera.cx], [0.0, camera.f, camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def build_pinhole_K(camera: CameraIntrinsics) -> np.ndarray:
    """
    The undistorted-pinhole intrinsic matrix used consistently by training, holdout eval, and
    `render.py`. SIMPLE_RADIAL's single k1 term is small here, so `cv2.getOptimalNewCameraMatrix`
    with alpha=0 (crop to valid pixels, same size) keeps focal length/principal point close to
    the original - we reuse the same K for simplicity and because the valid region covers
    virtually the whole frame at this k1 magnitude.
    """
    return build_distorted_K(camera)


def undistort_image(image: np.ndarray, camera: CameraIntrinsics) -> np.ndarray:
    K = build_distorted_K(camera)
    dist = np.array([camera.k1, 0.0, 0.0, 0.0], dtype=np.float64)  # SIMPLE_RADIAL: only k1
    return cv2.undistort(image, K, dist, newCameraMatrix=K)


def undistort_directory(image_dir: str, camera: CameraIntrinsics, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(".png"):
            continue
        src_path = os.path.join(image_dir, fname)
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        undistorted = undistort_image(img, camera)
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, undistorted)
        written.append(out_path)
    return written


def undistort_all(image_dirs: Iterable[str], camera: CameraIntrinsics, output_dir: str) -> List[str]:
    written = []
    for image_dir in image_dirs:
        written.extend(undistort_directory(image_dir, camera, output_dir))
    return written


def main():
    import argparse

    from src.colmap_io.reconstructor import PycolmapReconstructor

    parser = argparse.ArgumentParser(description="Undistort all UAV images to a pinhole convention")
    parser.add_argument("--colmap-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--unlabeled-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_dir = os.getenv("CONTEST_DATASET_DIR", os.path.join(project_root, "data", "Contest Dataset"))
    colmap_dir = args.colmap_dir or os.path.join(dataset_dir, "camera_parameters")
    images_dir = args.images_dir or os.path.join(dataset_dir, "images")
    unlabeled_dir = args.unlabeled_dir or os.path.join(dataset_dir, "unlabeled_Images")
    output_dir = args.output_dir or os.path.join(project_root, "outputs", "undistorted_images")

    camera, _, _ = PycolmapReconstructor(colmap_dir).load()
    written = undistort_all([images_dir, unlabeled_dir], camera, output_dir)
    print(f"[undistort] wrote {len(written)} undistorted images to {output_dir}")


if __name__ == "__main__":
    main()
