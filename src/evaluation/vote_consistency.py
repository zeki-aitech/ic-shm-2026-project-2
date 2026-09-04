"""
Analyzes the raw multi-view semantic vote distributions behind the Gaussian Splatting semantic
warm-start (`src.colmap_io.semantic_voting`), quantifying:
1. How consistent each class's plurality-winning vote is (mean/median vote share, fraction
   clearing an absolute majority).
2. How much `vote_majority_class`'s strict-majority rule for `stay_cable` actually changes
   relative to plain plurality voting, and where the reclassified points go.

These are the numbers the paper's Method section (3.4) cites to justify applying the
strict-majority rule only to `stay_cable` rather than to all five classes.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.colmap_io.semantic_voting import CLASS_NAMES, STAY_CABLE_CLASS_ID, vote_majority_class


def plain_plurality(labels: List[int]) -> int:
    """Plurality winner with no special-casing for any class (unlike `vote_majority_class`)."""
    if not labels:
        return 0
    return Counter(labels).most_common(1)[0][0]


@dataclass
class ClassVoteConsistency:
    n_points: int
    mean_vote_share: float
    median_vote_share: float
    frac_over_majority: float


@dataclass
class VoteConsistencyReport:
    n_points_with_votes: int
    per_class: Dict[int, ClassVoteConsistency]
    n_plurality_cable: int
    n_strict_cable: int
    n_reclassified_from_cable: int
    reclassified_destination_counts: Dict[int, int]

    def to_markdown(self) -> str:
        lines = [
            "# Multi-View Vote Consistency Analysis",
            "",
            f"Computed over {self.n_points_with_votes} sparse 3D points with at least one "
            "train-view observation (Method Section 3.4/3.6).",
            "",
            "## Per-class plurality-winner vote share",
            "| Class | # points won | mean share | median share | frac. clearing 50% |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
        for cid in sorted(self.per_class.keys()):
            c = self.per_class[cid]
            lines.append(
                f"| {CLASS_NAMES.get(cid, cid)} | {c.n_points} | {c.mean_vote_share:.3f} | "
                f"{c.median_vote_share:.3f} | {c.frac_over_majority:.3f} |"
            )
        lines += [
            "",
            "## Strict-majority rule vs. plain plurality (stay_cable only)",
            f"- Points labeled cable under plain plurality: {self.n_plurality_cable}",
            f"- Points labeled cable under the strict-majority rule: {self.n_strict_cable}",
            f"- Reclassified away from cable by the strict rule: "
            f"{self.n_reclassified_from_cable} "
            f"({self.n_reclassified_from_cable / max(1, self.n_plurality_cable) * 100:.1f}% "
            "of plurality-cable points)",
            "- Destination classes for reclassified points:",
        ]
        for cid, n in sorted(self.reclassified_destination_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {CLASS_NAMES.get(cid, cid)}: {n}")
        return "\n".join(lines)


def analyze_vote_consistency(observations: Dict[int, List[int]]) -> VoteConsistencyReport:
    """
    `observations`: p3d_id -> raw list of observed 2D mask labels, e.g. from
    `SemanticProjector.gather_observations(include_image_stems=train_ids)`.
    """
    vote_shares_by_class: Dict[int, List[float]] = {}
    over_majority_by_class: Dict[int, List[bool]] = {}
    plurality_result: Dict[int, int] = {}
    strict_result: Dict[int, int] = {}
    n_points_with_votes = 0

    for p3d_id, labels in observations.items():
        if not labels:
            continue
        n_points_with_votes += 1
        counts = Counter(labels)
        winner, wcount = counts.most_common(1)[0]
        share = wcount / len(labels)
        vote_shares_by_class.setdefault(winner, []).append(share)
        over_majority_by_class.setdefault(winner, []).append(share > 0.5)
        plurality_result[p3d_id] = winner
        strict_result[p3d_id] = vote_majority_class(labels)

    per_class: Dict[int, ClassVoteConsistency] = {}
    for cid, shares in vote_shares_by_class.items():
        shares_arr = np.array(shares)
        over = np.array(over_majority_by_class[cid])
        per_class[cid] = ClassVoteConsistency(
            n_points=len(shares_arr),
            mean_vote_share=float(shares_arr.mean()),
            median_vote_share=float(np.median(shares_arr)),
            frac_over_majority=float(over.mean()),
        )

    cable_under_plurality = {pid for pid, c in plurality_result.items() if c == STAY_CABLE_CLASS_ID}
    cable_under_strict = {pid for pid, c in strict_result.items() if c == STAY_CABLE_CLASS_ID}
    reclassified = cable_under_plurality - cable_under_strict
    destination_counts = Counter(strict_result[pid] for pid in reclassified)

    return VoteConsistencyReport(
        n_points_with_votes=n_points_with_votes,
        per_class=per_class,
        n_plurality_cable=len(cable_under_plurality),
        n_strict_cable=len(cable_under_strict),
        n_reclassified_from_cable=len(reclassified),
        reclassified_destination_counts=dict(destination_counts),
    )


def main():
    import argparse
    import os

    from src.colmap_io.reconstructor import PycolmapReconstructor
    from src.colmap_io.semantic_voting import SemanticProjector
    from src.evaluation.metrics import trajectory_interleaved_split

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(
        description="Analyze multi-view semantic vote consistency (paper Method Section 3.4)"
    )
    parser.add_argument("--colmap-dir", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--gt-masks-dir", default=None)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_dir = os.getenv("CONTEST_DATASET_DIR", os.path.join(project_root, "data", "Contest Dataset"))
    colmap_dir = args.colmap_dir or os.path.join(dataset_dir, "camera_parameters")
    images_dir = args.images_dir or os.path.join(dataset_dir, "images")
    gt_masks_dir = args.gt_masks_dir or os.path.join(project_root, "outputs", "gt_masks")
    output_path = args.output or os.path.join(project_root, "outputs", "eval", "vote_consistency_report.md")

    labeled_ids = sorted(
        os.path.splitext(f)[0] for f in os.listdir(images_dir)
        if f.lower().endswith(".png") and os.path.splitext(f)[0].isdigit()
    )
    train_ids, _holdout_ids = trajectory_interleaved_split(labeled_ids, args.holdout_ratio)

    camera, images, pts3d = PycolmapReconstructor(colmap_dir).load()

    class _CachedParser:
        def load(self):
            return camera, images, pts3d

    projector = SemanticProjector(colmap_dir, gt_masks_dir, parser=_CachedParser())
    observations = projector.gather_observations(include_image_stems=set(train_ids))

    report = analyze_vote_consistency(observations)
    print(report.to_markdown())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())
    print(f"\n[vote_consistency] wrote {output_path}")


if __name__ == "__main__":
    main()
