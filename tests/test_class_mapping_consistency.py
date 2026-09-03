import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.json_to_mask import CLASS_MAPPING
from src.colmap_io.semantic_voting import CLASS_NAMES as VOTING_CLASS_NAMES
from src.evaluation.metrics import CLASS_NAMES as METRICS_CLASS_NAMES


class TestClassMappingConsistency(unittest.TestCase):
    def test_all_class_maps_agree(self):
        id_to_name = {v: k for k, v in CLASS_MAPPING.items()}
        self.assertEqual(id_to_name, VOTING_CLASS_NAMES)
        self.assertEqual(id_to_name, METRICS_CLASS_NAMES)

    def test_five_classes_zero_indexed(self):
        self.assertEqual(set(CLASS_MAPPING.values()), {0, 1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()
