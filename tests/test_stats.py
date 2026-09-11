import math
import unittest

from quantpilot.stats import paired_delta_pct


class TestPairedDelta(unittest.TestCase):
    def test_constant_shift_has_zero_width_interval(self):
        base = [2.0, 2.3, 1.9, 2.1]
        pct, lo, hi = paired_delta_pct([b + 0.01 for b in base], base)
        self.assertAlmostEqual(pct, (math.exp(0.01) - 1) * 100)
        self.assertAlmostEqual(lo, pct)
        self.assertAlmostEqual(hi, pct)

    def test_text_noise_cancels_but_disagreement_widens(self):
        base = [1.5, 2.8, 2.0, 2.4, 1.7, 2.6]  # chunk difficulty varies a lot
        steady = [b + 0.01 for b in base]
        noisy = [b + d for b, d in zip(base, (0.04, -0.02, 0.04, -0.02, 0.04, -0.02))]
        _, lo_s, hi_s = paired_delta_pct(steady, base)
        _, lo_n, hi_n = paired_delta_pct(noisy, base)
        self.assertLess(hi_s - lo_s, 1e-9)
        self.assertGreater(hi_n - lo_n, 1.0)

    def test_unpairable_inputs_return_none(self):
        self.assertIsNone(paired_delta_pct(None, [1.0, 2.0]))
        self.assertIsNone(paired_delta_pct([1.0, 2.0], [1.0, 2.0, 3.0]))
        self.assertIsNone(paired_delta_pct([1.0], [1.0]))


if __name__ == "__main__":
    unittest.main()
