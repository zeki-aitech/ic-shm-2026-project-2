"""
Source-agnostic semantic point-cloud filtering.

Works on any ASCII PLY produced by SemanticProjector (ColmapParser sparse or
pycolmap GPU) — operates only on (xyz, rgb, class_ids) arrays.
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import open3d as o3d

from src.reconstruction.visualizer import read_ply_file, CLASS_NAMES

# Per-class statistical outlier removal (SOR) parameters.
# stay_cable is thinner/sparser — use looser thresholds to avoid over-culling.
SOR_PARAMS: Dict[int, Tuple[int, float]] = {
    1: (20, 2.0),   # deck
    2: (16, 2.5),   # stay_cable
    3: (20, 2.0),   # tower
    4: (20, 2.0),   # foundation
}

DECK_CLASS_ID = 1
STAY_CABLE_CLASS_ID = 2
TOWER_CLASS_ID = 3
FOUNDATION_CLASS_ID = 4
BACKGROUND_CLASS_ID = 0
MIN_CABLE_POINTS_FOR_FAN = 100
CABLE_FAN_CLUSTERS = 2
MIN_DECK_FOR_ENVELOPE = 50
MIN_CABLE_FOR_ENVELOPE = 20
MIN_DECK_FOR_CORE = 50
ALONG_MARGIN_FRAC = 0.05
LATERAL_MARGIN_FRAC = 0.40
TOWER_TOP_MARGIN = 0.3
DECK_CORE_K = 20
DECK_CORE_MAD_MULTIPLIER = 5.0


@dataclass
class FilterStats:
    """Per-stage point counts for logging and tests."""
    initial: int = 0
    after_drop_background: int = 0
    after_statistical: Dict[int, int] = field(default_factory=dict)
    after_deck_plane: int = 0
    final: int = 0
    removed_by_stage: Dict[str, int] = field(default_factory=dict)


def _count_by_class(class_ids: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(class_ids, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def drop_background(
    xyz: np.ndarray,
    rgb: np.ndarray,
    class_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove class 0 (background) points."""
    mask = class_ids != BACKGROUND_CLASS_ID
    return xyz[mask], rgb[mask], class_ids[mask]


def statistical_outlier_removal_per_class(
    xyz: np.ndarray,
    rgb: np.ndarray,
    class_ids: np.ndarray,
    sor_params: Dict[int, Tuple[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Apply Open3D statistical outlier removal independently per structural class.
    Returns filtered arrays and per-class removed counts.
    """
    if sor_params is None:
        sor_params = SOR_PARAMS

    keep_mask = np.zeros(len(class_ids), dtype=bool)
    removed_by_class: Dict[int, int] = {}

    for cid, (nb_neighbors, std_ratio) in sor_params.items():
        class_mask = class_ids == cid
        n_class = int(class_mask.sum())
        if n_class == 0:
            continue

        if n_class <= nb_neighbors:
            # Too few points for SOR — keep all
            keep_mask[class_mask] = True
            removed_by_class[cid] = 0
            continue

        pts = xyz[class_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        _, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        inlier_indices = np.asarray(inlier_indices, dtype=np.int64)

        class_indices = np.where(class_mask)[0]
        keep_mask[class_indices[inlier_indices]] = True
        removed_by_class[cid] = n_class - len(inlier_indices)

    return xyz[keep_mask], rgb[keep_mask], class_ids[keep_mask], removed_by_class


def fit_plane_pca(xyz: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a plane to points via PCA. Returns unit normal and offset d such that
    n·x + d = 0 for points on the plane (centroid lies on plane).
    """
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -float(np.dot(normal, centroid))
    return normal, d


def plane_residuals(xyz: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Absolute distance from each point to the plane n·x + d = 0."""
    return np.abs(xyz @ normal + d)


def filter_deck_plane(
    xyz: np.ndarray,
    rgb: np.ndarray,
    class_ids: np.ndarray,
    mad_multiplier: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Fit a plane to deck points and remove those with large residuals.

    Uses a two-pass fit: coarse PCA on all deck points, drop extreme residuals,
    then refit on inliers and apply median + mad_multiplier * MAD threshold.
    Other classes are unchanged.
    """
    deck_mask = class_ids == DECK_CLASS_ID
    n_deck = int(deck_mask.sum())
    if n_deck < 4:
        return xyz, rgb, class_ids, 0

    deck_xyz = xyz[deck_mask]
    deck_indices = np.where(deck_mask)[0]

    # Pass 1: coarse plane, drop extreme outliers (e.g. > 95th percentile residual)
    normal, d = fit_plane_pca(deck_xyz)
    residuals = plane_residuals(deck_xyz, normal, d)
    coarse_threshold = np.percentile(residuals, 95)
    coarse_inlier = residuals <= coarse_threshold
    if coarse_inlier.sum() < 4:
        coarse_inlier = np.ones(len(deck_xyz), dtype=bool)

    # Pass 2: refit on coarse inliers, MAD-based cut
    refit_xyz = deck_xyz[coarse_inlier]
    normal, d = fit_plane_pca(refit_xyz)
    residuals = plane_residuals(deck_xyz, normal, d)

    median = np.median(residuals[coarse_inlier])
    mad = np.median(np.abs(residuals[coarse_inlier] - median))
    if mad < 1e-9:
        mad = np.std(residuals[coarse_inlier])
    if mad < 1e-9:
        return xyz, rgb, class_ids, 0

    threshold = median + mad_multiplier * mad
    deck_inlier = residuals <= threshold

    keep_mask = np.ones(len(class_ids), dtype=bool)
    keep_mask[deck_indices[~deck_inlier]] = False

    removed = int((~deck_inlier).sum())
    return xyz[keep_mask], rgb[keep_mask], class_ids[keep_mask], removed


def _kmeans_xy(xy: np.ndarray, k: int, max_iter: int = 50, seed: int = 42) -> np.ndarray:
    """Simple k-means on 2D coordinates; returns cluster label per point."""
    rng = np.random.default_rng(seed)
    n = len(xy)
    if n <= k:
        return np.arange(n, dtype=np.int32) % k

    centroids = xy[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int32)

    for _ in range(max_iter):
        dists = np.linalg.norm(xy[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1).astype(np.int32)
        new_centroids = np.array([xy[labels == j].mean(axis=0) if (labels == j).any()
                                  else centroids[j] for j in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels


def filter_cable_fan_planes(
    xyz: np.ndarray,
    rgb: np.ndarray,
    class_ids: np.ndarray,
    mad_multiplier: float = 3.0,
    ransac_distance: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Keep stay_cable points near one of two fan planes (k-means on XY + RANSAC).

    Cable-stayed bridges have two tower fans. Points far from their cluster plane
    (e.g. background seen through filled mask polygons) are removed.
    """
    cable_mask = class_ids == STAY_CABLE_CLASS_ID
    n_cable = int(cable_mask.sum())
    if n_cable < MIN_CABLE_POINTS_FOR_FAN:
        return xyz, rgb, class_ids, 0

    cable_xyz = xyz[cable_mask]
    cable_indices = np.where(cable_mask)[0]
    xy = cable_xyz[:, :2]

    cluster_labels = _kmeans_xy(xy, k=CABLE_FAN_CLUSTERS)
    keep_cable = np.zeros(n_cable, dtype=bool)

    for cluster_id in range(CABLE_FAN_CLUSTERS):
        cmask = cluster_labels == cluster_id
        if cmask.sum() < 4:
            keep_cable[cmask] = True
            continue

        cluster_pts = cable_xyz[cmask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_pts.astype(np.float64))

        try:
            plane_model, _ = pcd.segment_plane(
                distance_threshold=ransac_distance,
                ransac_n=3,
                num_iterations=1000,
            )
        except Exception:
            keep_cable[cmask] = True
            continue

        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        residuals = np.abs(cluster_pts @ normal + d)

        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        if mad < 1e-9:
            mad = float(np.std(residuals))
        if mad < 1e-9:
            keep_cable[cmask] = True
            continue

        threshold = median + mad_multiplier * mad
        inlier = residuals <= threshold
        cluster_idx = np.where(cmask)[0]
        keep_cable[cluster_idx[inlier]] = True

    keep_mask = np.ones(len(class_ids), dtype=bool)
    keep_mask[cable_indices[~keep_cable]] = False
    removed = int((~keep_cable).sum())
    return xyz[keep_mask], rgb[keep_mask], class_ids[keep_mask], removed


def estimate_up_from_reconstruction(model_path: str) -> np.ndarray:
    """
    Estimate world "up" (gravity) direction from COLMAP camera poses.

    Drone imagery is shot with near-zero roll, so the world direction of each
    image's up axis (-R^T @ [0,1,0]) clusters tightly around true vertical.
    Requires pycolmap; `model_path` points to a COLMAP model directory.
    """
    import pycolmap

    rec = pycolmap.Reconstruction(model_path)
    ups = []
    for img in rec.images.values():
        R = img.cam_from_world().rotation.matrix()
        ups.append(-R.T @ np.array([0.0, 1.0, 0.0]))
    up = np.mean(ups, axis=0)
    return up / (np.linalg.norm(up) + 1e-12)


def build_bridge_frame(
    deck_xyz: np.ndarray,
    up_hint: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build bridge-local frame from deck points.

    Vertical v comes from `up_hint` when given (e.g. camera-derived gravity —
    strongly recommended: the deck-plane PCA normal has an arbitrary sign and
    can be badly tilted on noisy decks). Without a hint, v falls back to the
    deck-plane normal and the caller is responsible for orienting its sign.

    Returns origin (deck centroid), longitudinal u, vertical v, lateral w.
    """
    origin = deck_xyz.mean(axis=0)

    if up_hint is not None:
        v = np.asarray(up_hint, dtype=np.float64)
    else:
        v, _ = fit_plane_pca(deck_xyz)
    v = v / (np.linalg.norm(v) + 1e-12)

    centered = deck_xyz - origin
    on_plane = centered - np.outer(centered @ v, v)
    _, _, vh = np.linalg.svd(on_plane, full_matrices=False)
    u = vh[0]
    u = u / (np.linalg.norm(u) + 1e-12)

    w = np.cross(u, v)
    w = w / (np.linalg.norm(w) + 1e-12)
    u = np.cross(v, w)
    u = u / (np.linalg.norm(u) + 1e-12)

    return origin, u, v, w


def to_bridge_coords(
    xyz: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world XYZ into bridge (along, height, lateral) coordinates."""
    rel = xyz - origin
    return rel @ u, rel @ v, rel @ w


def filter_deck_core_density(
    xyz: np.ndarray,
    rgb: np.ndarray,
    class_ids: np.ndarray,
    k: int = DECK_CORE_K,
    mad_multiplier: float = DECK_CORE_MAD_MULTIPLIER,
    up_hint: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Remove coplanar but spatially sparse deck points.

    Deck plane filtering only cuts residual-to-plane outliers. Scattered deck
    labels that still lie near the plane (far along / lateral of the dense
    roadway) are removed by k-NN density in the bridge (along, lateral) plane.
    Other classes are unchanged.
    """
    from scipy.spatial import cKDTree

    deck_mask = class_ids == DECK_CLASS_ID
    n_deck = int(deck_mask.sum())
    if n_deck < MIN_DECK_FOR_CORE or n_deck <= k:
        return xyz, rgb, class_ids, 0

    deck_xyz = xyz[deck_mask]
    deck_indices = np.where(deck_mask)[0]

    origin, u, v, w = build_bridge_frame(deck_xyz, up_hint=up_hint)
    if up_hint is None:
        # Prefer foundation / tower to orient vertical when no camera up is given.
        foundation_mask = class_ids == FOUNDATION_CLASS_ID
        tower_mask = class_ids == TOWER_CLASS_ID
        deck_h_med = float(np.median((deck_xyz - origin) @ v))
        flip = False
        if foundation_mask.sum() >= 10:
            flip = float(np.median((xyz[foundation_mask] - origin) @ v)) > deck_h_med
        elif tower_mask.any():
            t_h = (xyz[tower_mask] - origin) @ v
            above = float(np.percentile(t_h, 99)) - deck_h_med
            below = deck_h_med - float(np.percentile(t_h, 1))
            flip = below > above
        if flip:
            v = -v
            w = -w

    along, _, lat = to_bridge_coords(deck_xyz, origin, u, v, w)
    xy = np.column_stack([along, lat])
    tree = cKDTree(xy)
    nn_dist = tree.query(xy, k=k + 1)[0][:, -1]

    median = float(np.median(nn_dist))
    mad = float(np.median(np.abs(nn_dist - median)))
    if mad < 1e-9:
        mad = float(np.std(nn_dist))
    if mad < 1e-9:
        return xyz, rgb, class_ids, 0

    threshold = median + mad_multiplier * mad
    deck_inlier = nn_dist <= threshold

    keep_mask = np.ones(len(class_ids), dtype=bool)
    keep_mask[deck_indices[~deck_inlier]] = False
    removed = int((~deck_inlier).sum())
    return xyz[keep_mask], rgb[keep_mask], class_ids[keep_mask], removed


def filter_cable_structural_envelope(
    xyz: np.ndarray,
    rgb: np.ndarray,
    class_ids: np.ndarray,
    along_margin_frac: float = ALONG_MARGIN_FRAC,
    lateral_margin_frac: float = LATERAL_MARGIN_FRAC,
    tower_top_margin: float = TOWER_TOP_MARGIN,
    up_hint: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Keep stay_cable points inside the structural envelope defined by deck and tower.

    Uses bridge-local frame: longitudinal from deck PCA, vertical from `up_hint`
    (camera-derived gravity, preferred) or the deck-plane normal. Without a hint
    the normal's sign is disambiguated using foundation (must sit below deck) or
    tower (must extend above deck). Removes cable points below deck, above tower
    top, or outside deck length / corridor.
    """
    deck_mask = class_ids == DECK_CLASS_ID
    tower_mask = class_ids == TOWER_CLASS_ID
    cable_mask = class_ids == STAY_CABLE_CLASS_ID
    foundation_mask = class_ids == FOUNDATION_CLASS_ID

    n_deck = int(deck_mask.sum())
    n_cable = int(cable_mask.sum())
    if n_deck < MIN_DECK_FOR_ENVELOPE or n_cable < MIN_CABLE_FOR_ENVELOPE:
        return xyz, rgb, class_ids, 0

    deck_xyz = xyz[deck_mask]
    tower_xyz = xyz[tower_mask] if tower_mask.any() else deck_xyz
    cable_xyz = xyz[cable_mask]
    cable_indices = np.where(cable_mask)[0]

    origin, u, v, w = build_bridge_frame(deck_xyz, up_hint=up_hint)

    if up_hint is None:
        # SVD normal has arbitrary sign — orient v so "up" is physically up.
        deck_h_med = float(np.median((deck_xyz - origin) @ v))
        flip = False
        if foundation_mask.sum() >= 10:
            found_h_med = float(np.median((xyz[foundation_mask] - origin) @ v))
            flip = found_h_med > deck_h_med
        elif tower_mask.any():
            t_h = (tower_xyz - origin) @ v
            above = float(np.percentile(t_h, 99)) - deck_h_med
            below = deck_h_med - float(np.percentile(t_h, 1))
            flip = below > above
        if flip:
            v = -v
            w = -w  # keep the frame right-handed (u unchanged)

    d_along, d_height, d_lat = to_bridge_coords(deck_xyz, origin, u, v, w)
    t_along, t_height, t_lat = to_bridge_coords(tower_xyz, origin, u, v, w)
    c_along, c_height, c_lat = to_bridge_coords(cable_xyz, origin, u, v, w)

    along_lo = np.percentile(d_along, 1)
    along_hi = np.percentile(d_along, 99)
    along_span = max(along_hi - along_lo, 1e-6)
    along_lo -= along_margin_frac * along_span
    along_hi += along_margin_frac * along_span

    lat_ref = np.concatenate([d_lat, t_lat])
    lat_lo = np.percentile(lat_ref, 1)
    lat_hi = np.percentile(lat_ref, 99)
    lat_span = max(lat_hi - lat_lo, 1e-6)
    lat_lo -= lateral_margin_frac * lat_span
    lat_hi += lateral_margin_frac * lat_span

    deck_surface = float(np.median(d_height))
    deck_mad = float(np.median(np.abs(d_height - deck_surface)))
    if deck_mad < 1e-9:
        deck_mad = float(np.std(d_height))
    epsilon = max(deck_mad, 0.05)
    height_lo = deck_surface - epsilon
    height_hi = float(np.percentile(t_height, 99)) + tower_top_margin

    in_along = (c_along >= along_lo) & (c_along <= along_hi)
    in_lat = (c_lat >= lat_lo) & (c_lat <= lat_hi)
    in_height = (c_height > height_lo) & (c_height < height_hi)
    keep_cable = in_along & in_lat & in_height

    keep_mask = np.ones(len(class_ids), dtype=bool)
    keep_mask[cable_indices[~keep_cable]] = False
    removed = int((~keep_cable).sum())
    return xyz[keep_mask], rgb[keep_mask], class_ids[keep_mask], removed


def filter_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    class_ids: np.ndarray,
    remove_background: bool = True,
    apply_statistical: bool = True,
    apply_deck_plane: bool = True,
    apply_deck_core: bool = True,
    apply_cable_envelope: bool = True,
    apply_cable_fan: bool = False,
    sor_params: Dict[int, Tuple[int, float]] = None,
    deck_mad_multiplier: float = 3.0,
    deck_core_mad_multiplier: float = DECK_CORE_MAD_MULTIPLIER,
    cable_fan_mad_multiplier: float = 3.0,
    up_hint: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FilterStats]:
    """
    Run the full filter pipeline on semantic point cloud arrays.

    Stages (in order):
      1. Drop background (class 0)
      2. Per-class statistical outlier removal
      3. Deck plane residual filter
      4. Deck core density filter (along/lateral)
      5. Stay-cable structural envelope filter
      6. Stay-cable fan plane filter (optional, off by default)
    """
    stats = FilterStats(initial=len(class_ids))

    if remove_background:
        n_before = len(class_ids)
        xyz, rgb, class_ids = drop_background(xyz, rgb, class_ids)
        stats.after_drop_background = len(class_ids)
        stats.removed_by_stage["background"] = n_before - len(class_ids)
    else:
        stats.after_drop_background = len(class_ids)

    if apply_statistical and len(class_ids) > 0:
        n_before = len(class_ids)
        xyz, rgb, class_ids, removed = statistical_outlier_removal_per_class(
            xyz, rgb, class_ids, sor_params=sor_params,
        )
        stats.after_statistical = removed
        stats.removed_by_stage["statistical"] = n_before - len(class_ids)
    else:
        stats.after_statistical = {}

    if apply_deck_plane and len(class_ids) > 0:
        n_before = len(class_ids)
        xyz, rgb, class_ids, deck_removed = filter_deck_plane(
            xyz, rgb, class_ids, mad_multiplier=deck_mad_multiplier,
        )
        stats.after_deck_plane = len(class_ids)
        stats.removed_by_stage["deck_plane"] = deck_removed
    else:
        stats.after_deck_plane = len(class_ids)

    if apply_deck_core and len(class_ids) > 0:
        xyz, rgb, class_ids, core_removed = filter_deck_core_density(
            xyz, rgb, class_ids,
            mad_multiplier=deck_core_mad_multiplier,
            up_hint=up_hint,
        )
        stats.removed_by_stage["deck_core"] = core_removed
    else:
        stats.removed_by_stage.setdefault("deck_core", 0)

    if apply_cable_envelope and len(class_ids) > 0:
        xyz, rgb, class_ids, env_removed = filter_cable_structural_envelope(
            xyz, rgb, class_ids, up_hint=up_hint,
        )
        stats.removed_by_stage["cable_envelope"] = env_removed
    else:
        stats.removed_by_stage.setdefault("cable_envelope", 0)

    if apply_cable_fan and len(class_ids) > 0:
        xyz, rgb, class_ids, cable_removed = filter_cable_fan_planes(
            xyz, rgb, class_ids, mad_multiplier=cable_fan_mad_multiplier,
        )
        stats.removed_by_stage["cable_fan"] = cable_removed
    else:
        stats.removed_by_stage.setdefault("cable_fan", 0)

    stats.final = len(class_ids)
    return xyz, rgb, class_ids, stats


def write_ply_file(
    path: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    class_ids: np.ndarray,
) -> str:
    """Write ASCII PLY with x, y, z, red, green, blue, class_id."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(xyz)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int class_id\n")
        f.write("end_header\n")

        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            cid = int(class_ids[i])
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {cid}\n")

    return path


def print_filter_report(
    before_counts: Dict[int, int],
    after_counts: Dict[int, int],
    stats: FilterStats,
) -> None:
    """Print before/after per-class counts and stage removals."""
    print("\n--- Filter Report ---")
    print(f"Initial points: {stats.initial:,}")
    if "background" in stats.removed_by_stage:
        print(f"  Removed background: {stats.removed_by_stage['background']:,}")
    if "statistical" in stats.removed_by_stage:
        print(f"  Removed (statistical SOR): {stats.removed_by_stage['statistical']:,}")
        for cid, n in sorted(stats.after_statistical.items()):
            name = CLASS_NAMES.get(cid, f"class_{cid}")
            print(f"    {name}: {n:,} removed")
    if "deck_plane" in stats.removed_by_stage:
        print(f"  Removed (deck plane): {stats.removed_by_stage['deck_plane']:,}")
    if "deck_core" in stats.removed_by_stage:
        print(f"  Removed (deck core density): {stats.removed_by_stage['deck_core']:,}")
    if "cable_envelope" in stats.removed_by_stage:
        print(f"  Removed (cable envelope): {stats.removed_by_stage['cable_envelope']:,}")
    if "cable_fan" in stats.removed_by_stage:
        print(f"  Removed (cable fan): {stats.removed_by_stage['cable_fan']:,}")
    print(f"Final points: {stats.final:,}")

    print("\nPer-class counts (before -> after):")
    all_classes = sorted(set(before_counts) | set(after_counts))
    for cid in all_classes:
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        b = before_counts.get(cid, 0)
        a = after_counts.get(cid, 0)
        print(f"  [{cid}] {name:12s}: {b:6,} -> {a:6,}")


def filter_ply_file(
    input_path: str,
    output_path: str,
    remove_background: bool = True,
    apply_statistical: bool = True,
    apply_deck_plane: bool = True,
    apply_deck_core: bool = True,
    apply_cable_envelope: bool = True,
    apply_cable_fan: bool = False,
    up_hint: np.ndarray = None,
) -> FilterStats:
    """Load PLY, filter, write output, print report."""
    xyz, rgb, class_ids = read_ply_file(input_path)
    before_counts = _count_by_class(class_ids)

    xyz, rgb, class_ids, stats = filter_point_cloud(
        xyz, rgb, class_ids,
        remove_background=remove_background,
        apply_statistical=apply_statistical,
        apply_deck_plane=apply_deck_plane,
        apply_deck_core=apply_deck_core,
        apply_cable_envelope=apply_cable_envelope,
        apply_cable_fan=apply_cable_fan,
        up_hint=up_hint,
    )
    after_counts = _count_by_class(class_ids)

    write_ply_file(output_path, xyz, rgb, class_ids)
    print(f"Filtered {input_path}")
    print(f"  -> {output_path} ({stats.final:,} points)")
    print_filter_report(before_counts, after_counts, stats)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter semantic point cloud PLY (drop BG, per-class SOR, deck plane)",
    )
    parser.add_argument("--input", "-i", required=True, help="Input ASCII PLY path")
    parser.add_argument("--output", "-o", required=True, help="Output filtered PLY path")
    parser.add_argument(
        "--keep-background", action="store_true",
        help="Do not drop background class (default: drop)",
    )
    parser.add_argument(
        "--no-statistical", action="store_true",
        help="Skip per-class statistical outlier removal",
    )
    parser.add_argument(
        "--no-deck-plane", action="store_true",
        help="Skip deck plane residual filter",
    )
    parser.add_argument(
        "--no-deck-core", action="store_true",
        help="Skip deck along/lateral density core filter",
    )
    parser.add_argument(
        "--no-cable-envelope", action="store_true",
        help="Skip stay_cable structural envelope filter",
    )
    parser.add_argument(
        "--cable-fan", action="store_true",
        help="Enable stay_cable fan plane filter (off by default)",
    )
    parser.add_argument(
        "--up", type=str, default=None,
        help="World up vector 'x,y,z' for the bridge frame (overrides --colmap-model)",
    )
    parser.add_argument(
        "--colmap-model", type=str, default=None,
        help="COLMAP model dir to estimate up from camera poses (needs pycolmap)",
    )
    args = parser.parse_args()

    up_hint = None
    if args.up:
        up_hint = np.array([float(x) for x in args.up.split(",")], dtype=np.float64)
    elif args.colmap_model:
        up_hint = estimate_up_from_reconstruction(args.colmap_model)
        print(f"Estimated up from cameras: {np.round(up_hint, 3)}")

    filter_ply_file(
        args.input,
        args.output,
        remove_background=not args.keep_background,
        apply_statistical=not args.no_statistical,
        apply_deck_plane=not args.no_deck_plane,
        apply_deck_core=not args.no_deck_core,
        apply_cable_envelope=not args.no_cable_envelope,
        apply_cable_fan=args.cable_fan,
        up_hint=up_hint,
    )


if __name__ == "__main__":
    main()
