import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.evaluation.paper_statistics import mask_coverage, observation_summary


def test_coverage_pools_pixels_instead_of_averaging_image_percentages():
    with tempfile.TemporaryDirectory() as directory:
        a, b = Path(directory) / "a.png", Path(directory) / "b.png"
        Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(a)
        Image.fromarray(np.zeros((1, 1), dtype=np.uint8)).save(b)
        result = mask_coverage([a, b])
        assert result["deck"]["percent"] == 80
        assert result["background"]["pixels"] == 1


def test_rgb_track_length_differs_from_semantic_vote_count():
    observations = {1: [1, 1], 2: [1], 3: []}
    points = {1: SimpleNamespace(image_ids=[1, 2, 3, 4]),
              2: SimpleNamespace(image_ids=[1, 2]),
              3: SimpleNamespace(image_ids=[3, 4])}
    result = observation_summary(observations, points)
    assert result["deck"]["mean_rgb_observations"] == 3
    assert result["deck"]["mean_semantic_votes"] == 1.5
    assert result["background"]["points_with_labeled_observations"] == 0


def test_class_assignment_uses_training_tie_break():
    observations = {1: [0, 2]}
    result = observation_summary(observations, {1: SimpleNamespace(image_ids=[1, 2])})
    assert result["stay_cable"]["points_with_labeled_observations"] == 1
