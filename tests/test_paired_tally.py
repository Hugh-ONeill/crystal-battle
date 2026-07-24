"""Paired tally: pair completion, dispense-order prefix, discordant counts."""

import unittest

from showdown.paired_tally import paired_counts


class TestPairedCounts(unittest.TestCase):
    def test_discordant_and_concordant(self):
        # pair 1: A wins, B loses -> discordant A
        # pair 2: both lose      -> concordant, unscored
        # pair 3: A loses, B wins -> discordant B
        out = {1: True, 2: False, 3: False, 4: False, 5: False, 6: True}
        n_a, n_b, prefix, complete, w_a, l_a, w_b, l_b = paired_counts(out)
        self.assertEqual((n_a, n_b), (1, 1))
        self.assertEqual((prefix, complete), (3, 3))
        self.assertEqual((w_a, l_a), (1, 2))
        self.assertEqual((w_b, l_b), (1, 2))

    def test_incomplete_pair_stalls_prefix(self):
        # game 4 (pair 2, arm B) missing: prefix stops at pair 1 even though
        # pair 3 is complete and discordant
        out = {1: True, 2: False, 3: True, 5: False, 6: True}
        n_a, n_b, prefix, complete, *_ = paired_counts(out)
        self.assertEqual(prefix, 1)
        self.assertEqual(complete, 2)      # pairs 1 and 3
        self.assertEqual((n_a, n_b), (1, 0))  # pair 3's discordance not counted

    def test_empty(self):
        self.assertEqual(paired_counts({}), (0, 0, 0, 0, 0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
