"""
2D-to-3D semantic back-projection via multi-view majority voting.

For every triangulated 3D point, samples the 2D semantic mask class at each observing camera's
pixel location and votes a single class per point. This is used to warm-start the Semantic
Gaussian Splatting model's per-Gaussian semantic logits (`src/gaussian_splatting/model.py`)
instead of random initialization.
"""
import os
import time
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

from src.colmap_io.models import Point3D
from src.colmap_io.reconstructor import PycolmapReconstructor


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
    """Back-projects 2D mask labels onto 3D sparse points using multi-view majority voting."""

    def __init__(self, colmap_dir: str, gt_masks_dir: str, parser=None):
        self.colmap_dir = colmap_dir
        self.gt_masks_dir = gt_masks_dir

        # Any object with load() -> (camera, images, points3d) works, e.g. PycolmapReconstructor.
        self.parser = parser if parser is not None else PycolmapReconstructor(colmap_dir)
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
        """Preloads all available semantic mask PNGs from `gt_masks_dir` into memory."""
        t0 = time.time()
        print(f"🔄 Preloading masks from '{self.gt_masks_dir}'...")
        if not os.path.exists(self.gt_masks_dir):
            raise FileNotFoundError(f"Masks directory not found at {self.gt_masks_dir}")

        mask_files = [f for f in os.listdir(self.gt_masks_dir) if f.endswith('.png')]
        for mf in mask_files:
            mpath = os.path.join(self.gt_masks_dir, mf)
            self.mask_cache[mpath] = np.array(Image.open(mpath), dtype=np.uint8)

        t1 = time.time()
        print(f"✅ Loaded {len(self.mask_cache)} mask images into cache in {t1-t0:.2f}s!")
        return len(self.mask_cache)

    def gather_observations(
        self, include_image_stems: Optional[set] = None
    ) -> Dict[int, List[int]]:
        """
        Loads the reconstruction (if not already loaded) and the mask cache (if empty), then
        returns the raw per-3D-point list of observed 2D mask labels - one entry per
        (image, pixel) observation, in observation order, before any voting is applied. This is
        the shared data-gathering step behind both `project()` (which applies
        `vote_majority_class`) and `src.evaluation.vote_consistency` (which analyzes the raw
        vote distributions directly).

        `include_image_stems`: if given, only observations from images whose filename stem
        (e.g. "001") is in this set are included - used to restrict analysis to the train-view
        split (Section 3.6), matching how the semantic warm-start itself is computed.
        """
        cam, images, self.pts3d = self.parser.load()
        if not self.mask_cache:
            self.preload_masks()

        observations: Dict[int, List[int]] = {}
        for p3d_id, pt3d in self.pts3d.items():
            observed_labels: List[int] = []
            for img_id, pt2d_idx in zip(pt3d.image_ids, pt3d.point2d_idxs):
                img_pose = images[img_id]
                if include_image_stems is not None:
                    stem = os.path.splitext(img_pose.name)[0]
                    if stem not in include_image_stems:
                        continue
                mpath = self._get_mask_path(img_pose.name)
                if mpath and mpath in self.mask_cache:
                    mask = self.mask_cache[mpath]
                    u, v, _ = img_pose.points2d[pt2d_idx]
                    x, y = int(round(u)), int(round(v))
                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                        observed_labels.append(int(mask[y, x]))
            observations[p3d_id] = observed_labels
        return observations

    def project(self) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
        """Executes 2D-to-3D back-projection using multi-view majority voting."""
        t0 = time.time()
        print("🔄 Executing 2D-to-3D Semantic Back-Projection...")
        observations = self.gather_observations()

        self.point_classes.clear()
        self.point_colors.clear()

        class_counts = Counter()

        for p3d_id, observed_labels in observations.items():
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
