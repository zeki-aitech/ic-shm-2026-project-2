"""
Renders Figure 6 (Section 5.3, optional): a static point-cloud "screenshot" of the trained
Gaussians, exported via `SemanticGaussianModel.export_semantic_splat_ply` and colored by each
Gaussian's predicted semantic class.

This is a plain point-cloud scatter (position + decoded degree-0 SH color), not an actual
alpha-blended splat render from an interactive viewer (SuperSplat, antimatter15/splat) - a
reasonable static stand-in when no interactive viewer is available, and described as such in the
figure's caption rather than overclaimed.
"""
import os
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)

from src.colmap_io.semantic_voting import CLASS_COLORS

SH_C0 = 0.28209479177387814
_CLASS_LUT = np.stack([CLASS_COLORS[c] for c in range(len(CLASS_COLORS))]).astype(np.float32) / 255.0
BACKGROUND_CLASS_ID = 0

_PLY_FIELDS = [
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
]


def load_ply_points_colors(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parses the binary little-endian 3DGS-format PLY written by `export_splats` (the exact
    field layout `_export_splat_ply_with_colors` produces). Returns `(xyz [N,3], rgb [N,3] in
    [0,1])`, decoding degree-0 spherical harmonics back to color."""
    with open(ply_path, "rb") as f:
        header = b""
        while b"end_header\n" not in header:
            header += f.read(4096)
        header_end = header.index(b"end_header\n") + len(b"end_header\n")
        header_text = header[:header_end].decode("ascii")

        n_vertices = None
        for line in header_text.splitlines():
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
        if n_vertices is None:
            raise ValueError(f"Could not find 'element vertex' in PLY header: {ply_path}")

        f.seek(header_end)
        n_fields = len(_PLY_FIELDS)
        raw = f.read(n_vertices * n_fields * 4)

    arr = np.frombuffer(raw, dtype="<f4").reshape(n_vertices, n_fields)
    field_idx = {name: i for i, name in enumerate(_PLY_FIELDS)}
    xyz = arr[:, [field_idx["x"], field_idx["y"], field_idx["z"]]]
    f_dc = arr[:, [field_idx["f_dc_0"], field_idx["f_dc_1"], field_idx["f_dc_2"]]]
    rgb = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)
    return xyz, rgb


def classify_by_nearest_class_color(rgb: np.ndarray) -> np.ndarray:
    """Maps each decoded (noisy, SH-roundtrip) RGB color to the nearest official class color
    (`src.colmap_io.semantic_voting.CLASS_COLORS`) by squared distance. Exact equality checks
    against a hardcoded gray value are unreliable here: e.g. the background color (128,128,128)
    is 0.502 in [0,1], not exactly 0.5, and SH decoding adds further float noise."""
    dists = ((rgb[:, None, :] - _CLASS_LUT[None, :, :]) ** 2).sum(axis=2)
    return dists.argmin(axis=1)


def render_splat_pointcloud_views(
    ply_path: str,
    output_path: str,
    max_points: int = 200_000,
    max_background_points: int = 40_000,
    views=((25, 15),),
    point_size: float = 3.5,
    trim_percentile: float = 1.0,
    seed: int = 0,
) -> str:
    xyz, rgb = load_ply_points_colors(ply_path)
    n_total = xyz.shape[0]

    # Drop the sparse outlier floaters typical of 3DGS training (per-axis percentile trim), so
    # the plotted view isn't dominated by empty space around a handful of stray points.
    lo = np.percentile(xyz, trim_percentile, axis=0)
    hi = np.percentile(xyz, 100 - trim_percentile, axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]

    # Classify by nearest official class color (not a hardcoded-gray equality check, which is
    # unreliable: (128,128,128)/255 = 0.502, not exactly 0.5, before SH round-trip noise even
    # enters the picture) and recolor with the canonical palette for clean, unambiguous colors.
    class_ids = classify_by_nearest_class_color(rgb)
    canonical_rgb = _CLASS_LUT[class_ids]
    is_bg = class_ids == BACKGROUND_CLASS_ID

    rng = np.random.default_rng(seed)

    struct_xyz, struct_rgb = xyz[~is_bg], canonical_rgb[~is_bg]
    if struct_xyz.shape[0] > max_points:
        idx = rng.choice(struct_xyz.shape[0], size=max_points, replace=False)
        struct_xyz, struct_rgb = struct_xyz[idx], struct_rgb[idx]

    # A light background subsample for spatial context (deck surroundings, sky/water extent),
    # capped low and drawn faint so it doesn't drown out the structural classes.
    bg_xyz = xyz[is_bg]
    if bg_xyz.shape[0] > max_background_points:
        idx = rng.choice(bg_xyz.shape[0], size=max_background_points, replace=False)
        bg_xyz = bg_xyz[idx]

    # COLMAP/OpenCV world convention (Y down, Z forward-ish): flip Y and Z for a more natural
    # "looking at the bridge from outside" plot orientation.
    def _flip(a):
        return a[:, 0], -a[:, 1], -a[:, 2]

    sx, sy, sz = _flip(struct_xyz)
    bx, by, bz = _flip(bg_xyz)

    # Zoom to the structural-class bounding box (padded) rather than the full scene, which is
    # dominated by a much wider, diffuse background point spread.
    pad = 0.08
    x_lo, x_hi = sx.min(), sx.max()
    y_lo, y_hi = sy.min(), sy.max()
    z_lo, z_hi = sz.min(), sz.max()
    x_pad, y_pad, z_pad = (x_hi - x_lo) * pad, (y_hi - y_lo) * pad, (z_hi - z_lo) * pad

    n = len(views)
    fig = plt.figure(figsize=(8.0 * n, 6.0), dpi=220)
    for i, (elev, azim) in enumerate(views):
        left = i / n
        ax = fig.add_axes([left, 0.0, 1 / n, 1.0], projection="3d")
        ax.scatter(bx, bz, by, c="#c9c9c9", s=point_size * 0.35,
                   linewidths=0, depthshade=False, alpha=0.25)
        ax.scatter(sx, sz, sy, c=struct_rgb, s=point_size,
                   linewidths=0, depthshade=False, alpha=1.0)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(z_lo - z_pad, z_hi + z_pad)
        ax.set_zlim(y_lo - y_pad, y_hi + y_pad)
        ax.set_box_aspect((x_hi - x_lo, z_hi - z_lo, y_hi - y_lo))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    _autocrop_whitespace(output_path)
    return output_path


def _autocrop_whitespace(image_path: str, padding: int = 12) -> None:
    """3D scatter plots leave large near-white margins around the actual (thin, wide) point
    cloud regardless of `bbox_inches='tight'`, since that only trims around the full 3D axes
    box, not the rendered content within it. Crops to the actual non-white content instead."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    non_white = np.any(arr < 250, axis=2)
    if not non_white.any():
        return
    ys, xs = np.where(non_white)
    y0, y1 = max(0, ys.min() - padding), min(arr.shape[0], ys.max() + padding)
    x0, x1 = max(0, xs.min() - padding), min(arr.shape[1], xs.max() + padding)
    img.crop((x0, y0, x1, y1)).save(image_path)


def main():
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(description="Render Figure 6 (splat point cloud screenshot)")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "paper", "figures", "fig6_splat_pointcloud.png"),
    )
    args = parser.parse_args()

    out = render_splat_pointcloud_views(args.ply, args.output, max_points=args.max_points)
    print(f"[plot_splat_pointcloud] wrote {out}")


if __name__ == "__main__":
    main()
