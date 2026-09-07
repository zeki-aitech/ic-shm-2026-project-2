import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.plot_pipeline_diagram import (
    plot_pipeline_diagram,
    plot_pipeline_diagram_horizontal,
)


class TestPlotPipelineDiagram(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "fig1.png")
            returned = plot_pipeline_diagram(out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_creates_missing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "nested", "dir", "fig1.png")
            plot_pipeline_diagram(nested)
            self.assertTrue(os.path.exists(nested))


class TestPlotPipelineDiagramHorizontal(unittest.TestCase):
    def test_writes_a_nonempty_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "fig1_horizontal.png")
            returned = plot_pipeline_diagram_horizontal(out_path)
            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)


if __name__ == "__main__":
    unittest.main()
