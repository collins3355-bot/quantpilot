import unittest

from quantpilot.fit import FitRow, compute_rows, recommend, render
from quantpilot.memory import ModelShape, kv_bytes_per_token, shape_from_metadata

GB = 1024**3

QWEN3_SHAPE = ModelShape(
    arch="qwen3", block_count=36, head_count_kv=8,
    key_length=128, value_length=128, context_length=40960,
)


class TestShape(unittest.TestCase):
    def test_explicit_key_value_lengths(self):
        meta = {
            "general.architecture": "qwen3",
            "qwen3.block_count": 36,
            "qwen3.attention.head_count": 32,
            "qwen3.attention.head_count_kv": 8,
            "qwen3.embedding_length": 4096,
            "qwen3.attention.key_length": 128,
            "qwen3.attention.value_length": 128,
        }
        shape = shape_from_metadata(meta)
        self.assertEqual((shape.key_length, shape.value_length), (128, 128))

    def test_head_dim_fallback(self):
        meta = {
            "general.architecture": "qwen2",
            "qwen2.block_count": 24,
            "qwen2.attention.head_count": 14,
            "qwen2.attention.head_count_kv": 2,
            "qwen2.embedding_length": 896,
        }
        shape = shape_from_metadata(meta)
        self.assertEqual((shape.key_length, shape.value_length), (64, 64))

    def test_kv_bytes_per_token(self):
        # 36 layers x 8 KV heads x (128+128) x 2 bytes = 147456
        self.assertEqual(kv_bytes_per_token(QWEN3_SHAPE, "f16"), 147456)
        self.assertAlmostEqual(
            kv_bytes_per_token(QWEN3_SHAPE, "q8_0"), 147456 / 2 * (34 / 32)
        )


def report(variants):
    return {
        "source": "/m/Qwen3-8B-BF16.gguf",
        "engine": "llama.cpp",
        "variants": variants,
    }


def variant(name, size_gb, dppl):
    return {
        "name": name, "size_bytes": int(size_gb * GB),
        "ppl_increase_pct": dppl, "path": f"/w/{name}.gguf",
    }


class TestFit(unittest.TestCase):
    def test_rows_and_recommendation(self):
        rep = report(
            [
                variant("baseline", 15.26, 0.0),
                variant("Q6_K", 6.26, 0.83),
                variant("Q4_K_M", 4.68, 2.43),
            ]
        )
        # 8 GB budget @ 8192 ctx: KV = 1.125 GB -> Q6_K totals ~7.4 GB (fits),
        # baseline ~16.4 GB (doesn't)
        rows = compute_rows(rep, QWEN3_SHAPE, 8192, "f16", 8 * GB)
        by_name = {r.name: r for r in rows}
        self.assertFalse(by_name["baseline"].fits)
        self.assertTrue(by_name["Q6_K"].fits)
        self.assertTrue(by_name["Q4_K_M"].fits)
        self.assertEqual(recommend(rows).name, "Q6_K")  # best quality that fits

    def test_nothing_fits(self):
        rows = compute_rows(
            report([variant("Q4_K_M", 4.68, 2.43)]), QWEN3_SHAPE, 8192, "f16", 1 * GB
        )
        self.assertIsNone(recommend(rows))
        md = render(
            report([variant("Q4_K_M", 4.68, 2.43)]), rows, QWEN3_SHAPE,
            8192, "f16", 8, 0.75, {"chip": "Apple M1 Max"},
        )
        self.assertIn("Nothing fits", md)

    def test_render_warns_past_declared_context(self):
        rep = report([variant("Q6_K", 6.26, 0.83)])
        rows = compute_rows(rep, QWEN3_SHAPE, 65536, "f16", 64 * GB)
        md = render(rep, rows, QWEN3_SHAPE, 65536, "f16", 64, 0.75, {})
        self.assertIn("declared maximum", md)


if __name__ == "__main__":
    unittest.main()
