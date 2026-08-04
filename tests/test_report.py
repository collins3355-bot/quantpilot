import json
import unittest
from pathlib import Path

from quantpilot.pipeline import BenchRun, Variant
from quantpilot.report import human_size, render_markdown, to_json

HARDWARE = {"chip": "Apple M1 Max", "memory_gb": 64, "os": "Darwin 25.4.0", "machine": "arm64"}


def make_run():
    baseline = Variant("baseline", Path("/m/f16.gguf"), 2 * 1024**3, 10.0, 0.1, 4000.0, 200.0)
    q4 = Variant("Q4_K_M", Path("/w/q4.gguf"), 1 * 1024**3, 10.05, 0.1, 5000.0, 300.0)
    return BenchRun(
        source=Path("/m/f16.gguf"),
        baseline=baseline,
        variants=[q4],
        corpus=Path("/d/wiki.test.raw"),
        chunks=32,
        budget_pct=1.0,
    )


class TestReport(unittest.TestCase):
    def test_human_size(self):
        self.assertEqual(human_size(2 * 1024**3), "2.00 GB")
        self.assertEqual(human_size(500 * 1024**2), "500 MB")

    def test_markdown_contains_rows_and_recommendation(self):
        md = render_markdown(make_run(), HARDWARE)
        self.assertIn("| baseline | 2.00 GB", md)
        self.assertIn("| Q4_K_M | 1.00 GB | -50% ", md)
        self.assertIn("**Q4_K_M**", md)
        self.assertIn("Apple M1 Max", md)

    def test_json_round_trips(self):
        payload = json.loads(to_json(make_run(), HARDWARE))
        self.assertEqual(payload["recommendation"], "Q4_K_M")
        self.assertEqual(len(payload["variants"]), 2)
        self.assertAlmostEqual(payload["variants"][1]["ppl_increase_pct"], 0.5)


if __name__ == "__main__":
    unittest.main()
