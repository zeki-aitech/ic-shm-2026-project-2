"""
pycolmap-based camera/pose loading and sparse triangulation for the contest COLMAP model.

The contest dataset ships `cameras.txt`/`images.txt` (poses + 2D observations) but no
`points3D.txt`, so `PycolmapReconstructor` rebuilds feature tracks from the 2D observations and
triangulates every track robustly with LO-RANSAC (`pycolmap.estimate_triangulation`, includes
cheirality and minimum triangulation-angle checks). This is the camera/geometry foundation the
Semantic Gaussian Splatting pipeline (`src/gaussian_splatting/`) is built on: it supplies the
per-image poses used for training/rendering and the initial sparse point cloud used to warm-start
Gaussian positions and colors.
"""
import os
import shutil
import tempfile
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pycolmap

from src.colmap_io.models import CameraIntrinsics, ImagePose, Point3D

CONTEST_MODEL_FILES = ["cameras.txt", "images.txt", "rigs.txt", "frames.txt"]


def load_contest_model(colmap_dir: str) -> pycolmap.Reconstruction:
    """
    Load the contest camera_parameters folder as a pycolmap.Reconstruction.

    The dataset does not include points3D.txt (only poses and 2D observations),
    while pycolmap.Reconstruction requires the file to exist. We stage the model
    in a temp dir with an empty points3D.txt; the 2D observations keep their
    point3D_id references, which is all we need to rebuild tracks.
    """
    tmp = tempfile.mkdtemp()
    try:
        for fname in CONTEST_MODEL_FILES:
            src = os.path.join(colmap_dir, fname)
            if not os.path.exists(src):
                raise FileNotFoundError(f"{fname} not found in {colmap_dir}")
            shutil.copy(src, tmp)
        open(os.path.join(tmp, "points3D.txt"), "w").close()
        return pycolmap.Reconstruction(tmp)
    finally:
        shutil.rmtree(tmp)


def build_tracks(rec: pycolmap.Reconstruction) -> Dict[int, list]:
    """Group 2D observations by point3D_id: id -> [(image_id, point2D_idx), ...]."""
    tracks: Dict[int, list] = defaultdict(list)
    for image_id, image in rec.images.items():
        for idx, p2d in enumerate(image.points2D):
            if p2d.has_point3D():
                tracks[p2d.point3D_id].append((image_id, idx))
    return tracks


def triangulate_tracks(
    rec: pycolmap.Reconstruction,
    min_track_length: int = 2,
    min_tri_angle_deg: float = 0.5,
) -> Dict[str, int]:
    """
    Triangulate all tracks with LO-RANSAC and insert the resulting 3D points
    (with their tracks) into the reconstruction. Returns counters.
    """
    options = pycolmap.EstimateTriangulationOptions()
    options.min_tri_angle = np.deg2rad(min_tri_angle_deg)

    poses = {img_id: img.cam_from_world() for img_id, img in rec.images.items()}
    stats = {"ok": 0, "rejected": 0, "too_short": 0}

    for p3d_id, obs in build_tracks(rec).items():
        if len(obs) < min_track_length:
            stats["too_short"] += 1
            continue

        points2d = np.array([rec.images[iid].points2D[idx].xy for iid, idx in obs])
        cams_from_world = [poses[iid] for iid, _ in obs]
        cameras = [rec.images[iid].camera for iid, _ in obs]

        result = pycolmap.estimate_triangulation(points2d, cams_from_world, cameras, options)
        if result is None:
            stats["rejected"] += 1
            continue

        track = pycolmap.Track()
        for iid, idx in obs:
            track.add_element(iid, idx)
        point = pycolmap.Point3D()
        point.xyz = result["xyz"]
        point.track = track
        rec.add_point3D_with_id(p3d_id, point)
        stats["ok"] += 1

    rec.update_point_3d_errors()
    return stats


def reconstruction_to_parser_data(
    rec: pycolmap.Reconstruction,
) -> Tuple[CameraIntrinsics, Dict[int, ImagePose], Dict[int, Point3D]]:
    """Convert a pycolmap.Reconstruction into the CameraIntrinsics/ImagePose/Point3D dataclasses."""
    cam_id = next(iter(rec.cameras))
    cam = rec.cameras[cam_id]
    params = list(cam.params)  # SIMPLE_RADIAL: [f, cx, cy, k1]
    k1 = params[3] if len(params) > 3 else 0.0
    camera = CameraIntrinsics(
        id=cam_id, model=cam.model_name, width=cam.width, height=cam.height,
        f=params[0], cx=params[1], cy=params[2], k1=k1,
    )

    images: Dict[int, ImagePose] = {}
    for image_id, image in rec.images.items():
        pose = image.cam_from_world()
        qx, qy, qz, qw = pose.rotation.quat  # pycolmap stores xyzw
        qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
        tvec = np.asarray(pose.translation, dtype=np.float64)
        P = pose.matrix()  # 3x4 [R | t]
        R, T = P[:, :3], P[:, 3:4]

        points2d = [
            (float(p2d.xy[0]), float(p2d.xy[1]),
             int(p2d.point3D_id) if p2d.has_point3D() else -1)
            for p2d in image.points2D
        ]
        images[image_id] = ImagePose(
            image_id=image_id, name=image.name, qvec=qvec, tvec=tvec,
            R=R, T=T, P=P, camera_id=image.camera_id, points2d=points2d,
        )

    points3d: Dict[int, Point3D] = {}
    for p3d_id, p3d in rec.points3D.items():
        elems = p3d.track.elements
        points3d[p3d_id] = Point3D(
            id=p3d_id,
            xyz=np.asarray(p3d.xyz, dtype=np.float64),
            image_ids=[e.image_id for e in elems],
            point2d_idxs=[e.point2D_idx for e in elems],
        )

    return camera, images, points3d


def filter_outliers_iqr(points3d: Dict[int, Point3D], iqr_multiplier: float = 3.0) -> int:
    """IQR distance-from-median outlier filter, mutates `points3d` in place. Returns removed count."""
    if not points3d:
        return 0
    all_xyz = np.array([pt.xyz for pt in points3d.values()])
    dists = np.linalg.norm(all_xyz - np.median(all_xyz, axis=0), axis=1)
    q1, q3 = np.percentile(dists, [25, 75])
    upper_fence = q3 + iqr_multiplier * (q3 - q1)

    to_remove = [pid for pid, d in zip(list(points3d.keys()), dists) if d > upper_fence]
    for pid in to_remove:
        del points3d[pid]
    return len(to_remove)


class PycolmapReconstructor:
    """
    Loads the contest COLMAP model and triangulates its sparse point cloud.

    Exposes a `load() -> (camera, images, points3d)` contract, so it can be passed to
    `SemanticProjector` via its `parser` argument.
    """

    def __init__(self, colmap_dir: str, iqr_multiplier: float = 3.0):
        self.colmap_dir = colmap_dir
        self.iqr_multiplier = iqr_multiplier
        self.reconstruction: pycolmap.Reconstruction = None
        self.camera: CameraIntrinsics = None
        self.images: Dict[int, ImagePose] = {}
        self.points3d: Dict[int, Point3D] = {}

    def load(self) -> Tuple[CameraIntrinsics, Dict[int, ImagePose], Dict[int, Point3D]]:
        t0 = time.time()
        print(f"[pycolmap] Loading contest model from '{self.colmap_dir}'...")
        self.reconstruction = load_contest_model(self.colmap_dir)
        print(f"[pycolmap] {self.reconstruction.num_images()} images loaded. "
              f"Triangulating tracks with LO-RANSAC...")

        stats = triangulate_tracks(self.reconstruction)
        errors = np.array([p.error for p in self.reconstruction.points3D.values()])
        print(f"[pycolmap] Triangulated {stats['ok']} points "
              f"({stats['rejected']} rejected, {stats['too_short']} too short). "
              f"Reprojection error: mean={errors.mean():.2f}px, median={np.median(errors):.2f}px")

        self.camera, self.images, self.points3d = reconstruction_to_parser_data(self.reconstruction)
        n_removed = filter_outliers_iqr(self.points3d, self.iqr_multiplier)
        print(f"[pycolmap] IQR outlier filter removed {n_removed} points. "
              f"Final: {len(self.points3d)} points in {time.time() - t0:.1f}s "
              f"(CUDA build: {pycolmap.has_cuda})")
        return self.camera, self.images, self.points3d


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATASET_DIR = os.getenv("CONTEST_DATASET_DIR", os.path.join(PROJECT_ROOT, "data", "Contest Dataset"))
    COLMAP_DIR = os.path.join(DATASET_DIR, "camera_parameters")

    reconstructor = PycolmapReconstructor(COLMAP_DIR)
    cam, imgs, pts3d = reconstructor.load()

    print("\n--- Summary ---")
    print(f"Camera: {cam.model} f={cam.f:.2f} pp=({cam.cx:.1f}, {cam.cy:.1f}) k1={cam.k1:.6f}")
    print(f"Images: {len(imgs)} | 3D points: {len(pts3d)}")
    sample = next(iter(pts3d.values()))
    print(f"Sample point {sample.id}: xyz={sample.xyz}, observed in {len(sample.image_ids)} views")
