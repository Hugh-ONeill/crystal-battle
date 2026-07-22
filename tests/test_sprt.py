"""SPRT gate math, checked against hand-computed values for the canonical
0.30-vs-0.40 question at alpha=beta=0.05: a win adds ln(4/3)=0.2877, a loss
adds ln(0.6/0.7)=-0.1542, and the bounds are +/-ln(19)=+/-2.9444."""

import math
import unittest

from showdown.sprt import bounds, expected_n, llr, verdict


class TestLlr(unittest.TestCase):
    def test_hand_computed_steps(self):
        self.assertAlmostEqual(llr(1, 0, 0.3, 0.4), math.log(4 / 3), places=10)
        self.assertAlmostEqual(llr(0, 1, 0.3, 0.4), math.log(0.6 / 0.7), places=10)
        self.assertAlmostEqual(llr(0, 0, 0.3, 0.4), 0.0)

    def test_symmetric_bounds(self):
        lo, hi = bounds(0.05, 0.05)
        self.assertAlmostEqual(hi, math.log(19), places=10)
        self.assertAlmostEqual(lo, -math.log(19), places=10)


class TestVerdict(unittest.TestCase):
    def test_h1_boundary_crossing(self):
        # 10-0 sits just under ln(19); the 11th win crosses it
        self.assertEqual(verdict(10, 0, 0.3, 0.4), "continue")
        self.assertEqual(verdict(11, 0, 0.3, 0.4), "accept-h1")
        # 11W-1L: 3.164 - 0.154 = 3.010 >= 2.944 still concludes
        self.assertEqual(verdict(11, 1, 0.3, 0.4), "accept-h1")

    def test_h0_boundary_crossing(self):
        # 19 straight losses: -2.930 > -2.944; the 20th crosses
        self.assertEqual(verdict(0, 19, 0.3, 0.4), "continue")
        self.assertEqual(verdict(0, 20, 0.3, 0.4), "accept-h0")

    def test_mixed_record_continues(self):
        # near the historical ~33% band: no conclusion at small n
        self.assertEqual(verdict(6, 12, 0.3, 0.4), "continue")

    def test_rigged_smoke_thresholds(self):
        # the harness smoke test uses p0=0.9 p1=0.99: one loss costs
        # ln(0.01/0.1) = -2.303, so two straight losses conclude
        self.assertEqual(verdict(0, 1, 0.9, 0.99), "continue")
        self.assertEqual(verdict(0, 2, 0.9, 0.99), "accept-h0")


class TestExpectedN(unittest.TestCase):
    def test_canonical_case(self):
        # hand-computed Wald approximation: E[n|H1]~118, E[n|H0]~123
        self.assertTrue(115 <= expected_n(0.3, 0.4) <= 130,
                        expected_n(0.3, 0.4))

    def test_wider_gap_needs_fewer_games(self):
        self.assertLess(expected_n(0.3, 0.5), expected_n(0.3, 0.4))
        self.assertLess(expected_n(0.3, 0.4), expected_n(0.3, 0.35))

    def test_looser_error_rates_need_fewer_games(self):
        self.assertLess(expected_n(0.3, 0.4, 0.10, 0.10),
                        expected_n(0.3, 0.4, 0.05, 0.05))


if __name__ == "__main__":
    unittest.main()
