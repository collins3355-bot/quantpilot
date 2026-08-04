import unittest
from pathlib import Path

from quantpilot.pipeline import BenchRun, Variant


def variant(name, size_gb, ppl):
    return Variant(
        name=name,
        path=Path(f"/tmp/{name}.gguf"),
        size_bytes=int(size_gb * 1024**3),
        ppl=ppl,
        ppl_err=0.1,
        prompt_tps=1000.0,
        generate_tps=100.0,
    )


def make_run(baseline, variants, budget_pct=1.0):
    return BenchRun(
        source=baseline.path,
        baseline=baseline,
        variants=variants,
        corpus=Path("/tmp/wiki.test.raw"),
        chunks=32,
        budget_pct=budget_pct,
    )


class TestRecommendation(unittest.TestCase):
    def test_picks_smallest_within_budget(self):
        baseline = variant("baseline", 2.0, 10.0)
        q8 = variant("Q8_0", 1.1, 10.02)   # +0.2%
        q4 = variant("Q4_K_M", 0.6, 10.08)  # +0.8% — inside 1% budget and smallest
        run = make_run(baseline, [q8, q4])
        self.assertEqual(run.recommendation().name, "Q4_K_M")

    def test_skips_variants_over_budget(self):
        baseline = variant("baseline", 2.0, 10.0)
        q8 = variant("Q8_0", 1.1, 10.05)   # +0.5%
        q4 = variant("Q4_K_M", 0.6, 10.30)  # +3.0% — over budget
        run = make_run(baseline, [q8, q4])
        self.assertEqual(run.recommendation().name, "Q8_0")

    def test_falls_back_to_baseline(self):
        baseline = variant("baseline", 2.0, 10.0)
        q4 = variant("Q4_K_M", 0.6, 12.0)  # +20% — way over budget
        run = make_run(baseline, [q4])
        self.assertEqual(run.recommendation().name, "baseline")

    def test_ppl_increase_pct(self):
        v = variant("Q4_K_M", 0.6, 10.5)
        self.assertAlmostEqual(v.ppl_increase_pct(10.0), 5.0)


if __name__ == "__main__":
    unittest.main()
