"""Tempo ledger: HP-token parsing, start-of-turn differential snapshots,
deficit-onset detection, and the hard-switch taxonomy (pivot follow-ups and
post-faint replacements must NOT count as hard switches or switch-in deaths)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from showdown.tempo_ledger import _frac, games_from_log, onset_turn

HEADER = [
    "|init|battle",
    "|player|p1|FPSpar1L1|266|",
    "|player|p2|CBGen9L1|169|",
    "|switch|p1a: Ting-Lu|Ting-Lu|100/100",
    "|switch|p2a: Pelipper|Pelipper, F|262/262",
]


def parse(lines):
    with TemporaryDirectory() as d:
        p = Path(d) / "x_ours.log"
        p.write_text("\n".join(lines) + "\n")
        return list(games_from_log(str(p)))


class TestFrac(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(_frac("0 fnt"), 0.0)
        self.assertEqual(_frac("131/262"), 0.5)
        self.assertEqual(_frac("66/100 psn"), 0.66)
        self.assertIsNone(_frac("[from] item: Leftovers"))


class TestLedger(unittest.TestCase):
    def test_diff_snapshot_and_onset(self):
        g, = parse(HEADER + [
            "|turn|1",
            "|move|p1a: Ting-Lu|Stealth Rock|",
            "|-damage|p2a: Pelipper|131/262",
            "|turn|2",                      # cb down 0.5 at start of t2
            "|-damage|p2a: Pelipper|0 fnt",
            "|faint|p2a: Pelipper",
            "|switch|p2a: Zapdos|Zapdos|301/301",
            "|turn|3",                      # cb down 1.0 -> onset
            "|win|FPSpar1L1",
        ])
        self.assertEqual(g["winner"], "fp")
        self.assertAlmostEqual(g["diffs"][2], -0.5)
        self.assertAlmostEqual(g["diffs"][3], -1.0)
        self.assertEqual(onset_turn(g, "cb", 1.0), 3)
        self.assertIsNone(onset_turn(g, "fp", 1.0))
        self.assertAlmostEqual(g["end_diff"], -1.0)

    def test_percent_and_absolute_hp_both_normalize(self):
        g, = parse(HEADER + [
            "|turn|1",
            "|-damage|p1a: Ting-Lu|50/100",
            "|-damage|p2a: Pelipper|131/262",
            "|turn|2",
            "|win|CBGen9L1",
        ])
        self.assertAlmostEqual(g["diffs"][2], 0.0)

    def test_pivot_and_replacement_are_not_hard_switches(self):
        g, = parse(HEADER + [
            "|turn|1",
            "|move|p2a: Pelipper|U-turn|p1a: Ting-Lu",
            "|switch|p2a: Zapdos|Zapdos|301/301",       # pivot follow-up
            "|move|p1a: Ting-Lu|Earthquake|p2a: Zapdos",
            "|-damage|p2a: Zapdos|0 fnt",
            "|faint|p2a: Zapdos",
            "|switch|p2a: Kingambit|Kingambit|100/100",  # replacement
            "|turn|2",
            "|switch|p2a: Pelipper|Pelipper, F|262/262",  # the real one
            "|turn|3",
            "|win|CBGen9L1",
        ])
        self.assertEqual(g["hard_sw"]["cb"], 1)
        self.assertEqual(g["dosi"]["cb"], 0)   # pivot mon death != switch-in death
        acts = [a for t in g["actions"] for a in g["actions"][t]
                if a[1] == "switch"]
        self.assertEqual(acts, [("cb", "switch", "Pelipper")])

    def test_died_on_hard_switch_in(self):
        g, = parse(HEADER + [
            "|turn|1",
            "|switch|p2a: Zapdos|Zapdos|301/301",
            "|move|p1a: Ting-Lu|Stone Edge|p2a: Zapdos",
            "|-damage|p2a: Zapdos|0 fnt",
            "|faint|p2a: Zapdos",
            "|turn|2",
            "|win|FPSpar1L1",
        ])
        self.assertEqual(g["hard_sw"]["cb"], 1)
        self.assertEqual(g["dosi"]["cb"], 1)
        self.assertEqual(g["dosi"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
