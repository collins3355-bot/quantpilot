import unittest

from quantpilot.engines import llamacpp


class TestPerplexityParsing(unittest.TestCase):
    def test_parses_final_estimate(self):
        output = (
            "perplexity: calculating perplexity over 32 chunks\n"
            "[1]6.4133,[2]7.1522,[3]7.0629\n"
            "Final estimate: PPL = 12.3456 +/- 0.21567\n"
        )
        ppl, err = llamacpp.parse_perplexity(output)
        self.assertAlmostEqual(ppl, 12.3456)
        self.assertAlmostEqual(err, 0.21567)

    def test_parses_without_error_bound(self):
        ppl, err = llamacpp.parse_perplexity("Final estimate: PPL = 9.5\n")
        self.assertAlmostEqual(ppl, 9.5)
        self.assertIsNone(err)

    def test_raises_when_missing(self):
        with self.assertRaises(llamacpp.EngineError):
            llamacpp.parse_perplexity("model failed to load")


class TestBenchParsing(unittest.TestCase):
    def test_extracts_prompt_and_generate_speeds(self):
        output = (
            "ggml_metal_init: found device\n"
            '[{"n_prompt": 512, "n_gen": 0, "avg_ts": 4131.7},'
            ' {"n_prompt": 0, "n_gen": 128, "avg_ts": 231.2}]\n'
        )
        speed = llamacpp.parse_bench_json(output)
        self.assertAlmostEqual(speed.prompt_tps, 4131.7)
        self.assertAlmostEqual(speed.generate_tps, 231.2)

    def test_raises_without_json(self):
        with self.assertRaises(llamacpp.EngineError):
            llamacpp.parse_bench_json("no json here")


if __name__ == "__main__":
    unittest.main()
