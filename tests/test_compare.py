import unittest

from quantpilot.compare import render_comparison


def report(engine, source, variants):
    return {
        "source": source,
        "engine": engine,
        "hardware": {"chip": "Apple M1 Max", "memory_gb": 64, "os": "Darwin"},
        "variants": variants,
    }


def variant(name, size_gb, ppl, dppl, tps=100.0):
    return {
        "name": name,
        "size_bytes": int(size_gb * 1024**3),
        "ppl": ppl,
        "ppl_increase_pct": dppl,
        "generate_tps": tps,
    }


GGUF = report(
    "llama.cpp",
    "/m/Qwen3-8B-BF16.gguf",
    [
        variant("baseline", 15.26, 8.77, 0.0),
        variant("Q6_K", 6.26, 8.85, 0.83),
        variant("Q4_K_M", 4.68, 8.99, 2.43),
    ],
)
MLX = report(
    "mlx-lm (group quantization)",
    "Qwen/Qwen3-8B",
    [
        variant("16-bit (baseline)", 15.2, 8.80, 0.0),
        variant("6-bit", 6.1, 8.86, 0.70),
        variant("4-bit", 4.3, 10.2, 15.9),
    ],
)


class TestComparison(unittest.TestCase):
    def test_rows_from_both_engines_sorted_by_size(self):
        md = render_comparison([GGUF, MLX])
        self.assertIn("| GGUF | Q6_K |", md)
        self.assertIn("| MLX | 6-bit |", md)
        # largest first: both baselines above both 6-bit rows
        self.assertLess(md.index("baseline"), md.index("Q6_K"))

    def test_frontier_picks_smallest_across_engines(self):
        md = render_comparison([GGUF, MLX])
        # within 1%: MLX 6-bit (6.1 GB) beats GGUF Q6_K (6.26 GB)
        self.assertIn("≤ +1% PPL: **MLX 6-bit**", md)
        # within 5%: GGUF Q4_K_M qualifies (2.43%), MLX 4-bit (15.9%) does not
        self.assertIn("≤ +5% PPL: **GGUF Q4_K_M**", md)

    def test_missing_engine_inferred_from_gguf_source(self):
        legacy = {k: v for k, v in GGUF.items() if k != "engine"}
        md = render_comparison([legacy])
        self.assertIn("| GGUF | Q6_K |", md)

    def test_deltas_are_vs_own_baseline(self):
        md = render_comparison([GGUF, MLX])
        self.assertIn("| +0.70% |", md)  # MLX 6-bit keeps its own-baseline delta
        self.assertIn("Δ columns are relative to each engine's own", md)


if __name__ == "__main__":
    unittest.main()
