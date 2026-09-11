import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.colmap_io.reconstructor import build_tracks


class _FakeP2D:
    def __init__(self, point3D_id):
        self._id = point3D_id

    def has_point3D(self):
        return self._id != -1

    @property
    def point3D_id(self):
        return self._id


class _FakeImage:
    def __init__(self, points2D):
        self.points2D = points2D


class _FakeReconstruction:
    def __init__(self, images):
        self.images = images


class TestBuildTracksExclusion(unittest.TestCase):
    """`exclude_image_ids` must keep an excluded image's observations out of every track, the
    same way the real pipeline keeps the held-out test split from influencing triangulation."""

    def setUp(self):
        # Point 100 is seen by images 1, 2, 3; point 101 only by images 1 and 3.
        self.rec = _FakeReconstruction({
            1: _FakeImage([_FakeP2D(100), _FakeP2D(101)]),
            2: _FakeImage([_FakeP2D(100), _FakeP2D(-1)]),  # -1: no triangulated point yet
            3: _FakeImage([_FakeP2D(100), _FakeP2D(101)]),
        })

    def test_no_exclusion_includes_every_observation(self):
        tracks = build_tracks(self.rec)
        self.assertEqual(sorted(tracks[100]), [(1, 0), (2, 0), (3, 0)])
        self.assertEqual(sorted(tracks[101]), [(1, 1), (3, 1)])

    def test_excluded_image_contributes_no_observation(self):
        tracks = build_tracks(self.rec, exclude_image_ids={3})
        self.assertEqual(sorted(tracks[100]), [(1, 0), (2, 0)])
        self.assertEqual(sorted(tracks[101]), [(1, 1)])
        seen_image_ids = {iid for obs in tracks.values() for iid, _ in obs}
        self.assertNotIn(3, seen_image_ids)

    def test_excluding_every_observer_drops_the_point_entirely(self):
        tracks = build_tracks(self.rec, exclude_image_ids={1, 3})
        self.assertEqual(sorted(tracks[100]), [(2, 0)])
        self.assertNotIn(101, tracks)  # only images 1 and 3 ever saw point 101

    def test_unobserved_points_never_appear(self):
        tracks = build_tracks(self.rec, exclude_image_ids={1, 2, 3})
        self.assertEqual(tracks, {})


if __name__ == "__main__":
    unittest.main()
