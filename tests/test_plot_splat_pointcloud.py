import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.plot_splat_pointcloud import (
    SH_C0,
    _PLY_FIELDS,
    load_ply_points_colors,
    render_splat_pointcloud_views,
)


def _write_synthetic_splat_ply(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """Writes a minimal binary PLY matching the exact field layout
    `_export_splat_ply_with_colors` produces, for testing the loader without gsplat."""
    n = xyz.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {name}\n" for name in _PLY_FIELDS)
        + "end_header\n"
    )
    f_dc = (rgb - 0.5) / SH_C0
    rest = {"opacity": 1.0, "scale_0": 0.1, "scale_1": 0.1, "scale_2": 0.1,
            "rot_0": 1.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0}
    arr = np.zeros((n, len(_PLY_FIELDS)), dtype="<f4")
    for i, name in enumerate(_PLY_FIELDS):
        if name in ("x", "y", "z"):
            arr[:, i] = xyz[:, "xyz".index(name)]
        elif name.startswith("f_dc_"):
            arr[:, i] = f_dc[:, int(name[-1])]
        else:
            arr[:, i] = rest[name]
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(arr.tobytes())


class TestLoadPlyPointsColors(unittest.TestCase):
    def test_round_trips_positions_and_colors(self):
        rng = np.random.default_rng(0)
        xyz = rng.uniform(-5, 5, size=(37, 3)).astype(np.float32)
        rgb = rng.uniform(0, 1, size=(37, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.ply")
            _write_synthetic_splat_ply(path, xyz, rgb)
            loaded_xyz, loaded_rgb = load_ply_points_colors(path)
            np.testing.assert_allclose(loaded_xyz, xyz, atol=1e-4)
            np.testing.assert_allclose(loaded_rgb, rgb, atol=1e-4)

    def test_colors_clipped_to_unit_range(self):
        xyz = np.zeros((3, 3), dtype=np.float32)
        rgb = np.array([[0.0, 0.5, 1.0]] * 3, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.ply")
            _write_synthetic_splat_ply(path, xyz, rgb)
            _, loaded_rgb = load_ply_points_colors(path)
            self.assertTrue(np.all(loaded_rgb >= 0.0) and np.all(loaded_rgb <= 1.0))


class TestRenderSplatPointcloudViews(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        rng = np.random.default_rng(0)
        xyz = rng.uniform(-5, 5, size=(500, 3)).astype(np.float32)
        rgb = rng.uniform(0, 1, size=(500, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            ply_path = os.path.join(tmp, "cloud.ply")
            _write_synthetic_splat_ply(ply_path, xyz, rgb)
            out_path = os.path.join(tmp, "fig6.png")
            returned = render_splat_pointcloud_views(ply_path, out_path, max_points=200)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)


if __name__ == "__main__":
    unittest.main()
