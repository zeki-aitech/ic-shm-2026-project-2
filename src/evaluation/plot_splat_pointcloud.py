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
import struct
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)

SH_C0 = 0.28209479177387814

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


def render_splat_pointcloud_views(
    ply_path: str,
    output_path: str,
    max_points: int = 200_000,
    views=((15, 10),),
    point_size: float = 1.8,
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

    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    # COLMAP/OpenCV world convention (Y down, Z forward-ish): flip Y and Z for a more natural
    # "looking at the bridge from outside" plot orientation.
    x, y, z = xyz[:, 0], -xyz[:, 1], -xyz[:, 2]

    # Draw background (gray) points small/first, structural-class points larger/on top, so the
    # bridge's structure is visually legible instead of drowned out by background point density.
    is_bg = np.all(np.isclose(rgb, 0.5, atol=1e-3), axis=1)

    panel_w = 9.5 if len(views) == 1 else 6.2
    panel_h = 5.0 if len(views) == 1 else 6.0
    fig = plt.figure(figsize=(panel_w * len(views), panel_h), dpi=200)
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        ax.scatter(x[is_bg], z[is_bg], y[is_bg], c=rgb[is_bg], s=point_size * 0.5,
                   linewidths=0, depthshade=False, alpha=0.35)
        ax.scatter(x[~is_bg], z[~is_bg], y[~is_bg], c=rgb[~is_bg], s=point_size * 1.8,
                   linewidths=0, depthshade=False, alpha=0.95)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(z.min(), z.max())
        ax.set_zlim(y.min(), y.max())
        ax.set_box_aspect((np.ptp(x), np.ptp(z), np.ptp(y)))

    fig.suptitle(
        f"Trained Gaussians colored by predicted semantic class "
        f"({xyz.shape[0]:,} of {n_total:,} shown)",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
