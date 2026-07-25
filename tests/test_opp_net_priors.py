"""Opponent-policy net -> s2 prior alignment invariants (no server needed)."""

import unittest

import numpy as np
import poke_engine as pe

from showdown.gen9_player import Gen9PokeEnginePlayer
from showdown.replay_to_policy_gen9 import load_policy_games


class _Stub:
    _net_aligned_s2 = Gen9PokeEnginePlayer._net_aligned_s2


def _state_and_options():
    for _w, _g, _m, recs in load_policy_games("showdown/gen9ou_policy_fp.pkl"):
        r = next((x for x in recs if x["s2"]), None)
        if r:
            st = pe.State.from_string(r["state"])
            warm = pe.monte_carlo_tree_search(
                pe.State.from_string(r["state"]), 1)
            return st, warm.side_two
    raise unittest.SkipTest("no fp policy corpus")


class TestNetS2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st, cls.opts = _state_and_options()
        cls.n = len(cls.opts)
        rng = np.random.default_rng(0)
        cls.dist = rng.dirichlet(np.ones(14))   # arbitrary 14-class dist

    def _s2(self, w):
        s = _Stub()
        s._opp_net_weight = w
        return s._net_aligned_s2(self.st, self.opts, self.dist)

    def test_normalized_and_floored(self):
        for w in (0.0, 0.5, 1.0):
            s2 = self._s2(w)
            self.assertAlmostEqual(sum(s2), 1.0, places=6)
            self.assertTrue(min(s2) > 0)          # every line keeps mass

    def test_w0_is_uniform(self):
        s2 = self._s2(0.0)
        self.assertTrue(all(abs(p - 1.0 / self.n) < 1e-9 for p in s2))

    def test_weight_sharpens_top(self):
        top = int(np.argmax(self._s2(1.0)))
        self.assertGreater(self._s2(1.0)[top], self._s2(0.5)[top])
        self.assertGreater(self._s2(0.5)[top], self._s2(0.0)[top])


if __name__ == "__main__":
    unittest.main()
