"""Hazard-cycle ledger: side-condition tracking, dirty-turn counting,
entry-cost attribution, SR-unpaid detection, and restick timing."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from showdown.hazard_cycle import games_from_log, restick

HEADER = [
    "|init|battle",
    "|player|p1|FPSpar1L1|266|",
    "|player|p2|CBGen9L1|169|",
    "|switch|p1a: Ting-Lu|Ting-Lu|100/100",
    "|switch|p2a: Corviknight|Corviknight, F|399/399",
]


def parse(lines):
    with TemporaryDirectory() as d:
        p = Path(d) / "x_ours.log"
        p.write_text("\n".join(lines) + "\n")
        return list(games_from_log(str(p)))


class TestHazardCycle(unittest.TestCase):
    def test_dirty_turns_entries_and_costs(self):
        g, = parse(HEADER + [
            "|turn|1",
            "|move|p1a: Ting-Lu|Stealth Rock|",
            "|-sidestart|p2: CBGen9L1|move: Stealth Rock",
            "|turn|2",                                     # cb dirty from t2
            "|switch|p2a: Gliscor|Gliscor, M|352/352",     # pays rocks
            "|-damage|p2a: Gliscor|308/352|[from] Stealth Rock",
            "|turn|3",
            "|switch|p2a: Corviknight|Corviknight, F|399/399",  # boots: unpaid
            "|turn|4",
            "|win|FPSpar1L1",
        ])
        self.assertEqual(g["dirty_turns"]["cb"], 3)   # t2, t3, t4
        self.assertEqual(g["dirty_turns"]["fp"], 0)
        self.assertEqual(g["entries"]["cb"], 2)
        self.assertEqual(g["sr_entries"]["cb"], 2)
        self.assertEqual(g["sr_paid"]["cb"], 1)       # Gliscor paid, Corv did not
        self.assertAlmostEqual(g["hazdmg"]["cb"], 44 / 352, places=4)
        self.assertAlmostEqual(g["hazdmg_mon"][("cb", "Gliscor")], 44 / 352,
                               places=4)

    def test_removal_clear_and_restick(self):
        g, = parse(HEADER + [
            "|turn|1",
            "|-sidestart|p2: CBGen9L1|move: Stealth Rock",
            "|turn|2",
            "|move|p2a: Corviknight|Defog|p1a: Ting-Lu",
            "|-sideend|p2: CBGen9L1|move: Stealth Rock|[from] move: Defog",
            "|turn|3",
            "|-sidestart|p2: CBGen9L1|move: Stealth Rock",   # re-set in 1 turn
            "|turn|4",
            "|win|CBGen9L1",
        ])
        self.assertEqual(g["removals"]["cb"], 1)
        self.assertEqual(len(g["cleared_at"]), 1)
        rs = restick([g])
        self.assertEqual(rs["cb"], [1, 1])
        # dirty on t2 (set t1), clean t3 snapshot? cleared during t2, re-set
        # during t3 -> dirty at t4 snapshot again
        self.assertEqual(g["dirty_turns"]["cb"], 2)   # t2 and t4

    def test_spikes_layers_and_sideend_clears_all(self):
        g, = parse(HEADER + [
            "|turn|1",
            "|-sidestart|p2: CBGen9L1|Spikes",
            "|turn|2",
            "|-sidestart|p2: CBGen9L1|Spikes",
            "|turn|3",
            "|-sideend|p2: CBGen9L1|Spikes",
            "|turn|4",
            "|win|CBGen9L1",
        ])
        self.assertEqual(g["dirty_turns"]["cb"], 2)   # t2, t3 (cleared in t3)
        self.assertEqual(g["sr_turns"]["cb"], 0)


if __name__ == "__main__":
    unittest.main()
