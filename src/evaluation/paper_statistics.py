"""Reproduce Section 5.2's mask coverage and sparse observation statistics.

Run: python -m src.evaluation.paper_statistics
Test mask areas use the undistorted evaluation masks. Observation counts distinguish
all retained RGB track observations from the labeled observations that cast semantic votes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.colmap_io.semantic_voting import (
    CLASS_NAMES,
    SemanticProjector,
    vote_majority_class,
)
from src.evaluation.metrics import train_val_test_split


def mask_coverage(mask_paths):
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for path in mask_paths:
        with Image.open(path) as image:
            mask = np.asarray(image)
        if mask.ndim != 2 or not np.isin(mask, list(CLASS_NAMES)).all():
            raise ValueError(f"Expected a class-ID mask with IDs 0–4: {path}")
        counts += np.bincount(mask.ravel(), minlength=len(CLASS_NAMES))
    if counts.sum() == 0:
        raise ValueError("No mask pixels were supplied")
    return {CLASS_NAMES[c]: {"pixels": int(n), "percent": float(100 * n / counts.sum())}
            for c, n in enumerate(counts)}


def observation_summary(observations, points):
    groups = {c: [] for c in CLASS_NAMES}
    for pid, labels in observations.items():
        if labels:
            groups[vote_majority_class(labels)].append(pid)
    return {
        CLASS_NAMES[c]: {
            "points_with_labeled_observations": len(ids),
            "mean_rgb_observations": float(np.mean([len(points[p].image_ids) for p in ids])) if ids else None,
            "mean_semantic_votes": float(np.mean([len(observations[p]) for p in ids])) if ids else None,
        }
        for c, ids in groups.items()
    }


def main():
    from src.colmap_io.reconstructor import PycolmapReconstructor

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=root / "data/Contest Dataset")
    parser.add_argument("--gt-masks-dir", type=Path, default=root / "outputs/gt_masks")
    parser.add_argument("--eval-masks-dir", type=Path, default=root / "outputs/undistorted_gt_masks")
    parser.add_argument("--output", type=Path, default=root / "outputs/eval/paper_statistics.json")
    args = parser.parse_args()
    labeled = sorted(p.stem for p in (args.dataset_dir / "images").glob("*.png"))
    train, val, test = train_val_test_split(labeled)
    colmap_dir = str(args.dataset_dir / "camera_parameters")
    reconstruction = PycolmapReconstructor(colmap_dir, exclude_image_names={f"{s}.png" for s in test})
    projector = SemanticProjector(colmap_dir, str(args.gt_masks_dir), parser=reconstruction)
    observations = projector.gather_observations(include_image_stems=set(train + val))
    excluded = {iid for iid, pose in reconstruction.images.items() if Path(pose.name).stem in test}
    if any(i in excluded for p in projector.pts3d.values() for i in p.image_ids):
        raise ValueError("Test observations occur in the reconstructed tracks")
    result = {
        "train_ids": train, "val_ids": val, "test_ids": test,
        "mask_coordinates": "undistorted",
        "sparse_points": len(projector.pts3d),
        "test_mask_coverage": mask_coverage(args.eval_masks_dir / f"{s}.png" for s in test),
        "observation_counts": observation_summary(observations, projector.pts3d),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_ids")}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
