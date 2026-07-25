import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy.spatial.transform as transform


@dataclass
class CameraIntrinsics:
    id: int
    model: str
    width: int
    height: int
    f: float
    cx: float
    cy: float
    k1: float


@dataclass
class ImagePose:
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
    id: int
    xyz: np.ndarray       # 3D coordinate (x, y, z) in world space
    image_ids: List[int]  # List of observing image IDs
    point2d_idxs: List[int] # List of 2D feature indices in respective images


class ColmapParser:
    """
    Parser for COLMAP Structure-from-Motion (SfM) camera parameters and 2D/3D data.
    Automatically handles camera intrinsics, image extrinsics, and DLT 3D triangulation.
    """

    def __init__(self, colmap_dir: str):
        self.colmap_dir = colmap_dir
        self.cameras_path = os.path.join(colmap_dir, "cameras.txt")
        self.images_path = os.path.join(colmap_dir, "images.txt")

        self.camera: Optional[CameraIntrinsics] = None
        self.images: Dict[int, ImagePose] = {}
        self.points3d: Dict[int, Point3D] = {}
        self.point_obs: Dict[int, List[Tuple[int, float, float, int]]] = {} # p3d_id -> list of (image_id, u, v, pt2d_idx)

    @staticmethod
    def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = qvec
        return transform.Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    def parse_cameras(self) -> CameraIntrinsics:
        if not os.path.exists(self.cameras_path):
            raise FileNotFoundError(f"cameras.txt not found at {self.cameras_path}")

        with open(self.cameras_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                model = parts[1]
                w = int(parts[2])
                h = int(parts[3])
                params = [float(p) for p in parts[4:]]

                # SIMPLE_RADIAL model: f, cx, cy, k1
                if model == "SIMPLE_RADIAL":
                    f_val, cx_val, cy_val, k1_val = params[0], params[1], params[2], params[3]
                elif model == "PINHOLE":
                    f_val, cx_val, cy_val, k1_val = params[0], params[2], params[3], 0.0
                elif model == "RADIAL":
                    f_val, cx_val, cy_val, k1_val = params[0], params[1], params[2], params[3]
                else:
                    # Default fallback
                    f_val, cx_val, cy_val, k1_val = params[0], params[1], params[2], 0.0

                self.camera = CameraIntrinsics(
                    id=cam_id, model=model, width=w, height=h,
                    f=f_val, cx=cx_val, cy=cy_val, k1=k1_val
                )
                break
        return self.camera

    def parse_images(self) -> Dict[int, ImagePose]:
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(f"images.txt not found at {self.images_path}")

        self.images.clear()
        self.point_obs.clear()

        with open(self.images_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        curr_img: Optional[ImagePose] = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()

            # Image header line has 10 fields: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            if len(parts) == 10 and parts[0].isdigit():
                img_id = int(parts[0])
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                cam_id = int(parts[8])
                img_name = parts[9]

                qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
                tvec = np.array([tx, ty, tz], dtype=np.float64)
                R = self.qvec2rotmat(qvec)
                T = tvec.reshape(3, 1)
                P = np.hstack([R, T])

                curr_img = ImagePose(
                    image_id=img_id, name=img_name,
                    qvec=qvec, tvec=tvec, R=R, T=T, P=P,
                    camera_id=cam_id, points2d=[]
                )
                self.images[img_id] = curr_img

            elif curr_img is not None:
                # 2D points observation line: X Y POINT3D_ID X Y POINT3D_ID ...
                pt2d_idx = 0
                for i in range(0, len(parts), 3):
                    u = float(parts[i])
                    v = float(parts[i+1])
                    p3d_id = int(parts[i+2])

                    curr_img.points2d.append((u, v, p3d_id))
                    if p3d_id != -1:
                        if p3d_id not in self.point_obs:
                            self.point_obs[p3d_id] = []
                        self.point_obs[p3d_id].append((curr_img.image_id, u, v, pt2d_idx))
                    pt2d_idx += 1

        return self.images

    def undistort_point(self, u_dist: float, v_dist: float) -> Tuple[float, float]:
        """
        Inverse simple radial distortion transformation to convert pixel coordinates
        to normalized undistorted camera coordinates.
        """
        if self.camera is None:
            raise ValueError("Camera intrinsics not parsed yet.")

        f = self.camera.f
        cx = self.camera.cx
        cy = self.camera.cy
        k1 = self.camera.k1

        x_dist = (u_dist - cx) / f
        y_dist = (v_dist - cy) / f

        if abs(k1) < 1e-12:
            return x_dist, y_dist

        x_u, y_u = x_dist, y_dist
        for _ in range(4):
            r2 = x_u**2 + y_u**2
            radial = 1.0 + k1 * r2
            x_u = x_dist / radial
            y_u = y_dist / radial
        return x_u, y_u

    def triangulate_all_points(self) -> Dict[int, Point3D]:
        """
        Computes 3D world coordinates for all 3D point IDs using DLT (Direct Linear Transform).
        """
        if not self.images or not self.camera:
            raise ValueError("Images and Camera parameters must be parsed before triangulation.")

        self.points3d.clear()

        for p3d_id, obs_list in self.point_obs.items():
            if len(obs_list) < 2:
                continue

            A = []
            img_ids = []
            pt2d_idxs = []

            for img_id, u, v, pt2d_idx in obs_list:
                x_u, y_u = self.undistort_point(u, v)
                P = self.images[img_id].P

                A.append(x_u * P[2, :] - P[0, :])
                A.append(y_u * P[2, :] - P[1, :])

                img_ids.append(img_id)
                pt2d_idxs.append(pt2d_idx)

            A_mat = np.array(A)
            _, _, Vt = np.linalg.svd(A_mat)
            X_homo = Vt[-1]

            if abs(X_homo[3]) < 1e-8:
                continue

            xyz = (X_homo[:3] / X_homo[3]).astype(np.float64)

            self.points3d[p3d_id] = Point3D(
                id=p3d_id,
                xyz=xyz,
                image_ids=img_ids,
                point2d_idxs=pt2d_idxs
            )

        return self.points3d

    def load(self) -> Tuple[CameraIntrinsics, Dict[int, ImagePose], Dict[int, Point3D]]:
        t0 = time.time()
        print(f"🔄 Parsing COLMAP data from '{self.colmap_dir}'...")
        self.parse_cameras()
        self.parse_images()
        print(f"  └─ Parsed Camera Model: {self.camera.model} ({self.camera.width}x{self.camera.height})")
        print(f"  └─ Parsed {len(self.images)} image poses.")
        print(f"  └─ Found {len(self.point_obs)} unique 3D Point IDs. Starting DLT 3D Triangulation...")
        
        self.triangulate_all_points()
        t1 = time.time()
        print(f"✅ COLMAP Parsing & Triangulation completed in {t1-t0:.2f}s! Total 3D points: {len(self.points3d)}")
        return self.camera, self.images, self.points3d


if __name__ == "__main__":
    COLMAP_DIR = "/workspaces/sfm_demo/data/Contest Dataset/camera_parameters"
    parser = ColmapParser(COLMAP_DIR)
    cam, imgs, pts3d = parser.load()
    
    print("\n--- Summary ---")
    print(f"Camera Focal Length: {cam.f:.2f}, Principal Point: ({cam.cx:.1f}, {cam.cy:.1f})")
    sample_img_id = next(iter(imgs))
    print(f"Sample Image ({imgs[sample_img_id].name}): Pose P shape={imgs[sample_img_id].P.shape}, 2D points count={len(imgs[sample_img_id].points2d)}")
    sample_p3d_id = next(iter(pts3d))
    print(f"Sample 3D Point ID {sample_p3d_id}: XYZ={pts3d[sample_p3d_id].xyz}, Observed in {len(pts3d[sample_p3d_id].image_ids)} images")
