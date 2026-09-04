import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.vote_consistency import (
    analyze_vote_consistency,
    plain_plurality,
)


class TestPlainPlurality(unittest.TestCase):
    def test_simple_majority(self):
        self.assertEqual(plain_plurality([1, 1, 0]), 1)

    def test_no_strict_majority_required(self):
        # Unlike vote_majority_class, cable (2) wins with plain plurality even at <=50%.
        self.assertEqual(plain_plurality([2, 2, 0, 0, 1]), 2)

    def test_empty(self):
        self.assertEqual(plain_plurality([]), 0)


class TestAnalyzeVoteConsistency(unittest.TestCase):
    def test_unanimous_votes_have_full_share(self):
        observations = {1: [1, 1, 1], 2: [3, 3]}
        report = analyze_vote_consistency(observations)
        self.assertEqual(report.n_points_with_votes, 2)
        self.assertAlmostEqual(report.per_class[1].mean_vote_share, 1.0)
        self.assertAlmostEqual(report.per_class[3].mean_vote_share, 1.0)
        self.assertEqual(report.per_class[1].n_points, 1)

    def test_points_with_no_observations_are_excluded(self):
        observations = {1: [1, 1], 2: []}
        report = analyze_vote_consistency(observations)
        self.assertEqual(report.n_points_with_votes, 1)

    def test_reclassification_from_plurality_cable_to_another_class(self):
        # 2/5 votes for cable (40%, the single largest block, so plain plurality picks cable)
        # but not an absolute majority -> vote_majority_class discards the cable votes and
        # falls back to plurality (a 3-way tie among the rest, broken by TIE_BREAK_PRIORITY
        # in favor of tower).
        observations = {1: [2, 2, 0, 1, 3]}
        report = analyze_vote_consistency(observations)
        self.assertEqual(report.n_plurality_cable, 1)
        self.assertEqual(report.n_strict_cable, 0)
        self.assertEqual(report.n_reclassified_from_cable, 1)
        self.assertEqual(report.reclassified_destination_counts.get(3), 1)

    def test_strict_majority_cable_is_not_reclassified(self):
        # 3/4 votes for cable (75%, clears the strict majority) -> stays cable under both rules.
        observations = {1: [2, 2, 2, 0]}
        report = analyze_vote_consistency(observations)
        self.assertEqual(report.n_plurality_cable, 1)
        self.assertEqual(report.n_strict_cable, 1)
        self.assertEqual(report.n_reclassified_from_cable, 0)

    def test_markdown_report_contains_key_numbers(self):
        observations = {1: [2, 2, 0, 0, 0], 2: [1, 1, 1]}
        report = analyze_vote_consistency(observations)
        md = report.to_markdown()
        self.assertIn("Multi-View Vote Consistency Analysis", md)
        self.assertIn("Reclassified away from cable", md)


if __name__ == "__main__":
    unittest.main()
