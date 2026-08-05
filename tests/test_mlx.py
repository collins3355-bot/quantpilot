import unittest

from quantpilot.engines import llamacpp, mlxlm


class TestRunnerOutputParsing(unittest.TestCase):
    def test_takes_last_json_line(self):
        output = (
            "Fetching 6 files: 100%\n"
            "some progress noise\n"
            '{"ppl": 13.9, "chunks": 32, "prompt_tps": 2100.5, "generate_tps": 180.0}\n'
        )
        result = mlxlm.parse_runner_output(output)
        self.assertAlmostEqual(result["ppl"], 13.9)
        self.assertAlmostEqual(result["generate_tps"], 180.0)

    def test_ignores_non_json_brace_lines(self):
        output = "{not json\n" '{"ppl": 1.0, "chunks": 1}\n'
        self.assertAlmostEqual(mlxlm.parse_runner_output(output)["ppl"], 1.0)

    def test_raises_without_json(self):
        with self.assertRaises(llamacpp.EngineError):
            mlxlm.parse_runner_output("no result here")

    def test_kld_keys_pass_through(self):
        output = '{"ppl": 8.8, "chunks": 32, "mean_kld": 0.015, "same_top_pct": 95.2}\n'
        result = mlxlm.parse_runner_output(output)
        self.assertAlmostEqual(result["mean_kld"], 0.015)
        self.assertAlmostEqual(result["same_top_pct"], 95.2)


if __name__ == "__main__":
    unittest.main()
