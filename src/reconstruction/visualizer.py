import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go


CLASS_NAMES = {
    0: "background",
    1: "deck",
    2: "stay_cable",
    3: "tower",
    4: "foundation"
}

CLASS_HEX_COLORS = {
    0: "#808080",  # Gray
    1: "#FF0000",  # Red
    2: "#00FFFF",  # Cyan
    3: "#00FF00",  # Green
    4: "#FFFF00"   # Yellow
}


def read_ply_file(ply_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads an ASCII PLY point cloud file with properties (x, y, z, red, green, blue, class_id).
    Returns (xyz, rgb, class_ids).
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found at {ply_path}")

    xyz_list = []
    rgb_list = []
    cid_list = []

    with open(ply_path, 'r', encoding='utf-8') as f:
        header = True
        for line in f:
            line = line.strip()
            if header:
                if line == "end_header":
                    header = False
                continue

            if not line:
                continue

            parts = line.split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
            cid = int(parts[6])

            xyz_list.append([x, y, z])
            rgb_list.append([r, g, b])
            cid_list.append(cid)

    return (
        np.array(xyz_list, dtype=np.float32),
        np.array(rgb_list, dtype=np.uint8),
        np.array(cid_list, dtype=np.int32)
    )


def create_interactive_3d_figure(
    xyz: np.ndarray,
    rgb: np.ndarray,
    class_ids: np.ndarray,
    point_size: float = 2.0,
    downsample_factor: int = 1
) -> go.Figure:
    """
    Creates a 3D interactive Plotly Scatter3d figure with per-class toggling capabilities.
    """
    if downsample_factor > 1:
        xyz = xyz[::downsample_factor]
        rgb = rgb[::downsample_factor]
        class_ids = class_ids[::downsample_factor]

    fig = go.Figure()

    unique_classes = np.unique(class_ids)

    for cid in sorted(unique_classes):
        mask = (class_ids == cid)
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        color_hex = CLASS_HEX_COLORS.get(cid, "#FFFFFF")

        pts = xyz[mask]
        pts_rgb = rgb[mask]

        # Use actual RGB values if available, or fall back to hex
        color_strings = [f"rgb({r},{g},{b})" for r, g, b in pts_rgb]

        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode='markers',
                name=f"[{cid}] {cname} ({len(pts)} pts)",
                marker=dict(
                    size=point_size,
                    color=color_strings,
                    opacity=0.85
                ),
                hovertemplate=f"<b>Class: {cname}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>"
            )
        )

    fig.update_layout(
        title=dict(
            text=f"🌉 IC-SHM 2026 — Semantic 3D Point Cloud ({len(xyz):,} points)",
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(title='X (m)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)"),
            yaxis=dict(title='Y (m)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)"),
            zaxis=dict(title='Z (m)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)"),
            aspectmode='data'
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0, 0, 0, 0.6)"
        )
    )

    return fig


if __name__ == "__main__":
    PLY_PATH = "/workspaces/sfm_demo/outputs/point_clouds/semantic_bridge_sparse.ply"
    xyz, rgb, cids = read_ply_file(PLY_PATH)
    print(f"Read {len(xyz)} points from {PLY_PATH}")
    fig = create_interactive_3d_figure(xyz, rgb, cids, point_size=2.0)
    out_html = "/workspaces/sfm_demo/outputs/point_clouds/semantic_bridge_interactive.html"
    fig.write_html(out_html)
    print(f"Saved interactive 3D HTML visualization to: {out_html}")
