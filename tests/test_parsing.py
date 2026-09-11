import math
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


class TestChunkNLL(unittest.TestCase):
    def test_recovers_per_chunk_nll_from_running_estimates(self):
        nll = llamacpp.parse_chunk_nll(
            "[1]6.4133,[2]7.1522,[3]7.0629\nFinal estimate: PPL = 7.0629 +/- 0.2\n"
        )
        self.assertEqual(len(nll), 3)
        self.assertAlmostEqual(nll[0], math.log(6.4133))
        # the per-chunk values average back to the final estimate
        self.assertAlmostEqual(sum(nll) / 3, math.log(7.0629))

    def test_empty_without_running_estimates(self):
        self.assertEqual(llamacpp.parse_chunk_nll("Final estimate: PPL = 9.5\n"), [])

    def test_ignores_hellaswag_style_brackets(self):
        self.assertEqual(llamacpp.parse_chunk_nll("16\t50.0%\t[27.9996%, 72.0004%]\n"), [])


class TestHellaSwagParsing(unittest.TestCase):
    SAMPLE = (
        "hellaswag: loading tasks\n"
        "14\t42.85714286%\t[21.3808%, 67.4094%]\n"
        "15\t46.66666667%\t[24.8095%, 69.8830%]\n"
        "16\t50.00000000%\t[27.9996%, 72.0004%]\n"
    )

    def test_takes_final_running_accuracy(self):
        acc, tasks = llamacpp.parse_hellaswag(self.SAMPLE)
        self.assertAlmostEqual(acc, 50.0)
        self.assertEqual(tasks, 16)

    def test_raises_when_missing(self):
        with self.assertRaises(llamacpp.EngineError):
            llamacpp.parse_hellaswag("model failed to load")


class TestKLDParsing(unittest.TestCase):
    SAMPLE = (
        "====== KL divergence statistics ======\n"
        "Mean    KLD:   0.028143 ±   0.001151\n"
        "Maximum KLD:   0.401573\n"
        "====== Token probability statistics ======\n"
        "Mean    Δp: -0.572 ± 0.124 %\n"
        "RMS Δp    :  3.993 ± 0.197 %\n"
        "Same top p: 91.961 ±  0.852 %\n"
    )

    def test_parses_mean_kld_and_top1(self):
        stats = llamacpp.parse_kld(self.SAMPLE)
        self.assertAlmostEqual(stats.mean_kld, 0.028143)
        self.assertAlmostEqual(stats.same_top_pct, 91.961)

    def test_top1_is_optional(self):
        stats = llamacpp.parse_kld("Mean    KLD:   0.005\n")
        self.assertAlmostEqual(stats.mean_kld, 0.005)
        self.assertIsNone(stats.same_top_pct)

    def test_raises_when_missing(self):
        with self.assertRaises(llamacpp.EngineError):
            llamacpp.parse_kld("no statistics here")


class TestBenchParsing(unittest.TestCase):
    def test_extracts_prompt_and_generate_speeds(self):
        output = (
            "ggml_metal_init: found device\n"
            '[{"n_prompt": 512, "n_gen": 0, "avg_ts": 4131.7, "stddev_ts": 22.1},'
            ' {"n_prompt": 0, "n_gen": 128, "avg_ts": 231.2, "stddev_ts": 11.7}]\n'
        )
        speed = llamacpp.parse_bench_json(output)
        self.assertAlmostEqual(speed.prompt_tps, 4131.7)
        self.assertAlmostEqual(speed.generate_tps, 231.2)
        self.assertAlmostEqual(speed.prompt_sd, 22.1)
        self.assertAlmostEqual(speed.generate_sd, 11.7)

    def test_spread_is_optional(self):
        speed = llamacpp.parse_bench_json('[{"n_prompt": 0, "n_gen": 128, "avg_ts": 50.0}]')
        self.assertAlmostEqual(speed.generate_tps, 50.0)
        self.assertIsNone(speed.generate_sd)

    def test_raises_without_json(self):
        with self.assertRaises(llamacpp.EngineError):
            llamacpp.parse_bench_json("no json here")


if __name__ == "__main__":
    unittest.main()
