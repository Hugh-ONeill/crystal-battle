"""Lead-EV blend: book habits sharpen the maximin, absence degrades to it."""

import unittest

from showdown.gen9_player import _lead_ev_blend

# rows = our leads, cols = their leads (order = opp_species)
MATRIX = [
    [0.2, 0.8, 0.5],   # great into col1, bad worst-case
    [0.4, 0.4, 0.4],   # flat, best worst-case
    [0.6, 0.1, 0.3],
]
SPECIES = ["gholdengo", "greattusk", "kingambit"]


class TestLeadEvBlend(unittest.TestCase):
    def test_no_book_is_pure_maximin(self):
        vals = _lead_ev_blend(MATRIX, SPECIES, None, 0)
        self.assertEqual(vals, [min(r) for r in MATRIX])
        vals = _lead_ev_blend(MATRIX, SPECIES, {"Pelipper": 5}, 10)
        self.assertEqual(vals, [min(r) for r in MATRIX])  # zero roster overlap

    def test_heavy_book_prefers_counterpick(self):
        # they always lead Great Tusk (col 1): row 0 becomes the pick
        counts = {"Great Tusk": 30}
        vals = _lead_ev_blend(MATRIX, SPECIES, counts, games=30)
        self.assertEqual(max(range(3), key=lambda i: vals[i]), 0)
        self.assertAlmostEqual(vals[0], 0.8, places=5)  # full weight, pure EV

    def test_partial_coverage_discounts(self):
        # half their observed leads are a mon not in today's roster
        counts = {"Great Tusk": 10, "Pelipper": 10}
        vals = _lead_ev_blend(MATRIX, SPECIES, counts, games=30)
        # weight = 1.0 * 0.5 -> row0 = 0.5*0.2 + 0.5*0.8 = 0.5
        self.assertAlmostEqual(vals[0], 0.5, places=5)

    def test_display_names_normalize(self):
        counts = {"Slowking-Galar": 4}
        vals = _lead_ev_blend(MATRIX, ["slowkinggalar", "pelipper", "amoonguss"],
                              counts, games=20)
        self.assertAlmostEqual(vals[2], 0.6, places=5)  # col 0 certain


if __name__ == "__main__":
    unittest.main()
