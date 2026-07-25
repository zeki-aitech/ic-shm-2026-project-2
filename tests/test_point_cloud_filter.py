"""Unit tests for semantic point cloud filtering."""
import os
import tempfile
import unittest

import numpy as np

from src.reconstruction.point_cloud_filter import (
    drop_background,
    filter_cable_fan_planes,
    filter_cable_structural_envelope,
    filter_deck_core_density,
    filter_deck_plane,
    filter_point_cloud,
    fit_plane_pca,
    plane_residuals,
    statistical_outlier_removal_per_class,
    write_ply_file,
)
from src.reconstruction.visualizer import read_ply_file


class TestDropBackground(unittest.TestCase):
    def test_removes_only_class_zero(self):
        xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        rgb = np.array([[128, 128, 128], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        cids = np.array([0, 1, 2], dtype=np.int32)

        fx, fr, fc = drop_background(xyz, rgb, cids)
        self.assertEqual(len(fx), 2)
        self.assertTrue(np.all(fc == np.array([1, 2])))


class TestStatisticalOutlierRemoval(unittest.TestCase):
    def test_removes_planted_outliers(self):
        rng = np.random.default_rng(42)
        cluster = rng.normal(0, 0.01, size=(200, 3)).astype(np.float32)
        outliers = np.array([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]], dtype=np.float32)
        xyz = np.vstack([cluster, outliers])
        rgb = np.full((len(xyz), 3), 255, dtype=np.uint8)
        cids = np.full(len(xyz), 1, dtype=np.int32)  # deck

        fx, _, fc, removed = statistical_outlier_removal_per_class(xyz, rgb, cids)
        self.assertGreater(removed.get(1, 0), 0)
        self.assertLess(len(fx), len(xyz))
        self.assertGreater(len(fx), 150)  # dense cluster mostly kept


class TestDeckPlaneFilter(unittest.TestCase):
    def test_removes_points_far_from_plane(self):
        rng = np.random.default_rng(0)
        inliers = np.column_stack([
            rng.uniform(0, 10, 200),
            rng.uniform(0, 2, 200),
            np.zeros(200, dtype=np.float32),
        ]).astype(np.float32)
        far = np.array([[5.0, 1.0, 8.0], [5.0, 1.0, -8.0]], dtype=np.float32)
        xyz = np.vstack([inliers, far])
        rgb = np.full((len(xyz), 3), 255, dtype=np.uint8)
        cids = np.full(len(xyz), 1, dtype=np.int32)

        fx, _, fc, removed = filter_deck_plane(xyz, rgb, cids, mad_multiplier=3.0)
        self.assertGreater(removed, 0)
        self.assertLess(len(fx), len(xyz))
        self.assertEqual(len(fx) + removed, len(xyz))

    def test_fit_plane_pca_horizontal(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
        normal, d = fit_plane_pca(pts)
        residuals = plane_residuals(pts, normal, d)
        self.assertTrue(np.allclose(residuals, 0, atol=1e-6))


class TestDeckCoreDensityFilter(unittest.TestCase):
    def test_removes_sparse_coplanar_outliers(self):
        rng = np.random.default_rng(0)
        # Dense roadway along X, narrow in Y, flat Z
        n = 400
        dense = np.column_stack([
            rng.uniform(0, 20, n),
            rng.normal(0, 0.05, n),
            np.zeros(n),
        ]).astype(np.float32)
        # Coplanar but far / sparse outliers (survive plane filter, fail density)
        sparse = np.array([
            [5.0, 8.0, 0.0],
            [10.0, -9.0, 0.0],
            [15.0, 12.0, 0.0],
            [-5.0, 0.0, 0.0],
            [25.0, 0.0, 0.0],
            [8.0, 6.0, 0.0],
            [12.0, -7.0, 0.0],
            [3.0, 10.0, 0.0],
        ], dtype=np.float32)
        # A few tower points so other classes stay untouched
        tower = np.array([[10.0, 0.0, z] for z in np.linspace(0, 5, 20)], dtype=np.float32)

        xyz = np.vstack([dense, sparse, tower])
        rgb = np.full((len(xyz), 3), 255, dtype=np.uint8)
        cids = np.concatenate([
            np.full(len(dense), 1, dtype=np.int32),
            np.full(len(sparse), 1, dtype=np.int32),
            np.full(len(tower), 3, dtype=np.int32),
        ])

        fx, _, fc, removed = filter_deck_core_density(
            xyz, rgb, cids, up_hint=np.array([0.0, 0.0, 1.0]),
        )
        self.assertGreaterEqual(removed, len(sparse) - 1)
        self.assertGreater((fc == 1).sum(), len(dense) * 0.9)
        self.assertEqual((fc == 3).sum(), len(tower))
        # Remaining deck should stay near the roadway corridor
        kept_deck = fx[fc == 1]
        self.assertLess(np.percentile(np.abs(kept_deck[:, 1]), 99), 1.0)


class TestCableEnvelopeFilter(unittest.TestCase):
    def test_removes_fakes_keeps_true_cables(self):
        rng = np.random.default_rng(0)
        n = 200
        # Deck along X at z=0
        deck_x = rng.uniform(0, 20, n)
        deck_y = rng.uniform(-1, 1, n) * 0.2
        deck = np.column_stack([deck_x, deck_y, np.zeros(n)]).astype(np.float32)

        # Two towers at x=5 and x=15, height up to z=10
        tower_pts = []
        for tx in [5.0, 15.0]:
            for z in np.linspace(0, 10, 80):
                tower_pts.append([tx, 0, z])
        tower = np.array(tower_pts, dtype=np.float32)

        # True cables between deck and mid-height inside span
        true_cables = []
        for tx in [5.0, 15.0]:
            for _ in range(60):
                t = rng.uniform(0, 1)
                x = tx + rng.uniform(-2, 2)
                y = rng.uniform(-0.5, 0.5)
                z = rng.uniform(0.5, 9.0)
                true_cables.append([x, y, z])
        true_cables = np.array(true_cables, dtype=np.float32)

        # Fakes: below deck and outside span
        fakes_below = rng.uniform(-2, 22, (40, 2))
        fakes_below = np.column_stack([
            fakes_below[:, 0], fakes_below[:, 1],
            rng.uniform(-5, -1, 40),
        ]).astype(np.float32)
        fakes_outside = np.column_stack([
            rng.uniform(-10, -2, 30),
            np.zeros(30),
            rng.uniform(0, 8, 30),
        ]).astype(np.float32)

        cable_xyz = np.vstack([true_cables, fakes_below, fakes_outside])
        rgb = np.full((len(cable_xyz), 3), 255, dtype=np.uint8)
        cids = np.full(len(cable_xyz), 2, dtype=np.int32)

        xyz = np.vstack([deck, tower, cable_xyz])
        rgb = np.vstack([
            np.full((len(deck), 3), 255, dtype=np.uint8),
            np.full((len(tower), 3), 0, dtype=np.uint8),
            rgb,
        ])
        cids = np.concatenate([
            np.full(len(deck), 1, dtype=np.int32),
            np.full(len(tower), 3, dtype=np.int32),
            cids,
        ])

        n_cable_before = len(cable_xyz)
        fx, _, fc, removed = filter_cable_structural_envelope(xyz, rgb, cids)
        n_cable_after = int((fc == 2).sum())

        self.assertGreater(removed, 50)
        self.assertLess(n_cable_after, n_cable_before)
        self.assertGreater(n_cable_after, len(true_cables) * 0.5)
        self.assertEqual((fc == 1).sum(), len(deck))
        self.assertEqual((fc == 3).sum(), len(tower))

    @staticmethod
    def _synthetic_scene(with_foundation=False):
        rng = np.random.default_rng(1)
        n = 200
        deck = np.column_stack([
            rng.uniform(0, 20, n), rng.uniform(-0.2, 0.2, n), np.zeros(n),
        ]).astype(np.float32)
        tower = np.array(
            [[tx, 0, z] for tx in [5.0, 15.0] for z in np.linspace(0, 10, 80)],
            dtype=np.float32,
        )
        true_cables = np.column_stack([
            rng.uniform(3, 17, 100), rng.uniform(-0.5, 0.5, 100), rng.uniform(0.5, 9.0, 100),
        ]).astype(np.float32)
        fakes_below = np.column_stack([
            rng.uniform(0, 20, 40), rng.uniform(-0.5, 0.5, 40), rng.uniform(-5, -1, 40),
        ]).astype(np.float32)

        parts = [deck, tower, true_cables, fakes_below]
        cids = [1] * len(deck) + [3] * len(tower) + [2] * (len(true_cables) + len(fakes_below))
        if with_foundation:
            foundation = np.column_stack([
                rng.uniform(4, 16, 50), rng.uniform(-0.5, 0.5, 50), rng.uniform(-3, -0.5, 50),
            ]).astype(np.float32)
            parts.append(foundation)
            cids += [4] * len(foundation)

        xyz = np.vstack(parts)
        rgb = np.full((len(xyz), 3), 255, dtype=np.uint8)
        return xyz, rgb, np.array(cids, dtype=np.int32), len(fakes_below)

    def test_up_hint_removes_below_deck_fakes(self):
        xyz, rgb, cids, n_fakes = self._synthetic_scene()
        _, _, fc, removed = filter_cable_structural_envelope(
            xyz, rgb, cids, up_hint=np.array([0.0, 0.0, 1.0]),
        )
        self.assertGreaterEqual(removed, n_fakes)
        self.assertGreater((fc == 2).sum(), 50)

    def test_foundation_orients_frame(self):
        xyz, rgb, cids, n_fakes = self._synthetic_scene(with_foundation=True)
        _, _, fc, removed = filter_cable_structural_envelope(xyz, rgb, cids)
        self.assertGreaterEqual(removed, n_fakes)
        self.assertGreater((fc == 2).sum(), 50)


class TestCableFanFilter(unittest.TestCase):
    def test_removes_far_outliers_from_two_fans(self):
        rng = np.random.default_rng(0)
        # Fan A: plane z = 0.1*x + 5, centered around x=-5
        n = 150
        xa = rng.uniform(-8, -2, n)
        ya = rng.uniform(0, 5, n)
        za = 0.1 * xa + 5 + rng.normal(0, 0.02, n)
        fan_a = np.column_stack([xa, ya, za]).astype(np.float32)

        # Fan B: plane z = -0.1*x + 5, centered around x=5
        xb = rng.uniform(2, 8, n)
        yb = rng.uniform(0, 5, n)
        zb = -0.1 * xb + 5 + rng.normal(0, 0.02, n)
        fan_b = np.column_stack([xb, yb, zb]).astype(np.float32)

        # Outliers far from both fans
        outliers = np.array([
            [-5, 2, 50], [5, 2, -50], [0, 2, 30],
        ], dtype=np.float32)

        cable_xyz = np.vstack([fan_a, fan_b, outliers])
        rgb = np.full((len(cable_xyz), 3), 255, dtype=np.uint8)
        cids = np.full(len(cable_xyz), 2, dtype=np.int32)
        # Add a few deck points so filter only touches cable
        deck = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        xyz = np.vstack([cable_xyz, deck])
        rgb = np.vstack([rgb, np.full((2, 3), 128, dtype=np.uint8)])
        cids = np.concatenate([cids, np.array([1, 1], dtype=np.int32)])

        fx, _, fc, removed = filter_cable_fan_planes(xyz, rgb, cids)
        self.assertGreater(removed, 0)
        self.assertLess((fc == 2).sum(), len(cable_xyz))
        # Deck untouched
        self.assertEqual((fc == 1).sum(), 2)
        # Most fan points kept
        self.assertGreater((fc == 2).sum(), 250)


class TestFilterPointCloud(unittest.TestCase):
    def test_full_pipeline_synthetic(self):
        rng = np.random.default_rng(0)
        n_bg, n_deck = 50, 100
        bg = rng.normal(0, 1, (n_bg, 3)).astype(np.float32)
        deck = rng.normal([5, 0, 0], 0.05, (n_deck, 3)).astype(np.float32)
        deck[-2:] = [100, 100, 100]  # outliers

        xyz = np.vstack([bg, deck])
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
        cids = np.array([0] * n_bg + [1] * n_deck, dtype=np.int32)

        fx, fr, fc, stats = filter_point_cloud(xyz, rgb, cids)
        self.assertEqual(stats.initial, n_bg + n_deck)
        self.assertNotIn(0, fc)
        self.assertLess(stats.final, n_deck)


class TestPlyRoundTrip(unittest.TestCase):
    def test_write_read_filter_roundtrip(self):
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        cids = np.array([1, 2], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.ply")
            write_ply_file(path, xyz, rgb, cids)
            self.assertTrue(os.path.exists(path))

            rx, rr, rc = read_ply_file(path)
            self.assertEqual(len(rx), 2)
            self.assertTrue(np.allclose(rx, xyz))
            self.assertTrue(np.array_equal(rc, cids))

            fx, fr, fc, stats = filter_point_cloud(
                rx, rr, rc, remove_background=False, apply_statistical=False, apply_deck_plane=False,
            )
            self.assertEqual(stats.final, 2)


if __name__ == "__main__":
    unittest.main()
