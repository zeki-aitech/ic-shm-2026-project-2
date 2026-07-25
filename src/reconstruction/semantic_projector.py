import os
import time
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

from src.reconstruction.colmap_parser import ColmapParser, Point3D


CLASS_NAMES = {
    0: "background",
    1: "deck",
    2: "stay_cable",
    3: "tower",
    4: "foundation"
}

CLASS_COLORS = {
    0: np.array([128, 128, 128], dtype=np.uint8),  # Gray
    1: np.array([255, 0, 0], dtype=np.uint8),      # Red
    2: np.array([0, 255, 255], dtype=np.uint8),    # Cyan
    3: np.array([0, 255, 0], dtype=np.uint8),      # Green
    4: np.array([255, 255, 0], dtype=np.uint8)     # Yellow
}

# Tie-breaking priority: thin/rare structures prioritized over large surfaces
TIE_BREAK_PRIORITY = {
    2: 5,  # stay_cable
    3: 4,  # tower
    4: 3,  # foundation
    1: 2,  # deck
    0: 1   # background
}


STAY_CABLE_CLASS_ID = 2
CABLE_ABSOLUTE_MAJORITY = 0.5  # strict: c / n must be > 0.5


def _plurality_with_tiebreak(counts: Counter) -> int:
    """Pick winner by plurality; break ties with TIE_BREAK_PRIORITY (excludes stay_cable)."""
    if not counts:
        return 0
    max_freq = max(counts.values())
    candidates = [cls_id for cls_id, freq in counts.items() if freq == max_freq]
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda c: TIE_BREAK_PRIORITY.get(c, 0), reverse=True)
    return candidates[0]


def vote_majority_class(labels: List[int]) -> int:
    """
    Determines the semantic class from multi-view 2D mask observations.

    stay_cable (2) is assigned only with absolute majority (>50%). Otherwise
    cable votes are ignored and plurality + tie-break applies among remaining classes.
    """
    if not labels:
        return 0

    counts = Counter(labels)
    n = len(labels)
    cable_count = counts.get(STAY_CABLE_CLASS_ID, 0)

    if cable_count / n > CABLE_ABSOLUTE_MAJORITY:
        return STAY_CABLE_CLASS_ID

    counts_no_cable = Counter({k: v for k, v in counts.items() if k != STAY_CABLE_CLASS_ID})
    return _plurality_with_tiebreak(counts_no_cable)


class SemanticProjector:
    """
    Back-projects 2D Ground-Truth PNG mask labels onto 3D sparse points
    using multi-view majority voting, and exports a colored semantic PLY point cloud.
    """

    def __init__(self, colmap_dir: str, gt_masks_dir: str, output_dir: str, parser=None):
        self.colmap_dir = colmap_dir
        self.gt_masks_dir = gt_masks_dir
        self.output_dir = output_dir

        # Any object with load() -> (camera, images, points3d) works,
        # e.g. PycolmapReconstructor or ReconstructionAdapter.
        self.parser = parser if parser is not None else ColmapParser(colmap_dir)
        self.mask_cache: Dict[str, np.ndarray] = {}

        self.point_classes: Dict[int, int] = {}       # p3d_id -> class_id
        self.point_colors: Dict[int, np.ndarray] = {}  # p3d_id -> [R, G, B]
        self.pts3d: Dict[int, Point3D] = {}

    def _get_mask_path(self, image_name: str) -> Optional[str]:
        stem = os.path.splitext(image_name)[0]
        if stem.isdigit():
            mask_filename = f"{int(stem):03d}.png"
        else:
            mask_filename = f"{stem}.png"

        mask_path = os.path.join(self.gt_masks_dir, mask_filename)
        if os.path.exists(mask_path):
            return mask_path
        return None

    def preload_masks(self) -> int:
        """
        Preloads all available Ground-Truth PNG masks into memory.
        """
        t0 = time.time()
        print(f"🔄 Preloading GT masks from '{self.gt_masks_dir}'...")
        if not os.path.exists(self.gt_masks_dir):
            raise FileNotFoundError(f"GT masks directory not found at {self.gt_masks_dir}")

        mask_files = [f for f in os.listdir(self.gt_masks_dir) if f.endswith('.png')]
        for mf in mask_files:
            mpath = os.path.join(self.gt_masks_dir, mf)
            self.mask_cache[mpath] = np.array(Image.open(mpath), dtype=np.uint8)

        t1 = time.time()
        print(f"✅ Loaded {len(self.mask_cache)} GT mask images into cache in {t1-t0:.2f}s!")
        return len(self.mask_cache)

    def project(self) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
        """
        Executes 2D-to-3D back-projection using multi-view majority voting.
        """
        t0 = time.time()
        print("🔄 Executing 2D-to-3D Semantic Back-Projection...")
        cam, images, self.pts3d = self.parser.load()

        if not self.mask_cache:
            self.preload_masks()

        self.point_classes.clear()
        self.point_colors.clear()

        class_counts = Counter()

        for p3d_id, pt3d in self.pts3d.items():
            observed_labels: List[int] = []

            for img_id, pt2d_idx in zip(pt3d.image_ids, pt3d.point2d_idxs):
                img_pose = images[img_id]
                mpath = self._get_mask_path(img_pose.name)

                if mpath and mpath in self.mask_cache:
                    mask = self.mask_cache[mpath]
                    u, v, _ = img_pose.points2d[pt2d_idx]

                    x = int(round(u))
                    y = int(round(v))

                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                        label = int(mask[y, x])
                        observed_labels.append(label)

            final_class = vote_majority_class(observed_labels)
            color = CLASS_COLORS.get(final_class, CLASS_COLORS[0])

            self.point_classes[p3d_id] = final_class
            self.point_colors[p3d_id] = color
            class_counts[final_class] += 1

        t1 = time.time()
        print(f"✅ Back-Projection complete in {t1-t0:.2f}s!")
        print("\n--- 3D Semantic Class Distribution ---")
        total_pts = len(self.pts3d)
        for cid in sorted(CLASS_NAMES.keys()):
            cnt = class_counts[cid]
            pct = (cnt / total_pts) * 100 if total_pts > 0 else 0
            cname = CLASS_NAMES[cid]
            print(f"  Class {cid} ({cname:12s}): {cnt:6d} points ({pct:5.2f}%)")

        return self.point_classes, self.point_colors

    def export_ply(self, output_path: str) -> str:
        """
        Exports the triangulated points and their 3D semantic colors to a standard ASCII PLY file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"🔄 Exporting Semantic Point Cloud to '{output_path}'...")

        num_points = len(self.pts3d)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property int class_id\n")
            f.write("end_header\n")

            for p3d_id, pt3d in self.pts3d.items():
                x, y, z = pt3d.xyz
                r, g, b = self.point_colors[p3d_id]
                cid = self.point_classes[p3d_id]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {cid}\n")

        print(f"✅ PLY export complete! File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        return output_path

    def run(self, output_filename: str = "semantic_bridge_sparse.ply") -> str:
        self.project()
        output_path = os.path.join(self.output_dir, output_filename)
        return self.export_ply(output_path)


if __name__ == "__main__":
    COLMAP_DIR = "/workspaces/sfm_demo/data/Contest Dataset/camera_parameters"
    GT_MASKS_DIR = "/workspaces/sfm_demo/outputs/gt_masks"
    OUTPUT_DIR = "/workspaces/sfm_demo/outputs/point_clouds"

    projector = SemanticProjector(COLMAP_DIR, GT_MASKS_DIR, OUTPUT_DIR)
    projector.run()
