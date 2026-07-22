"""Dispense-order tally: the SPRT must consume outcomes in the order games
were handed out, not the order they finished. The prefix stops at the first
in-flight index; no-decision games the lane moved past are skipped; decided
games beyond an in-flight index wait."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from showdown.dispense_tally import prefix_tally

def hdr(lane, g):
    return f"=== lane {lane} game {g}/18 team: G{g}_01_x (17:00:00) ===\n"

def win(lane):
    return f"INFO     Winner: CBGen9L{lane}\n"

def loss(lane):
    return f"INFO     Winner: FPSpar1L{lane}\n"


def write_logs(d, lanes):
    for lane, text in lanes.items():
        (Path(d) / f"t_L{lane}_foulplay.log").write_text(text)


class TestPrefixTally(unittest.TestCase):
    def test_contiguous_prefix_stops_at_inflight(self):
        with TemporaryDirectory() as d:
            write_logs(d, {
                # lane 1: G1 won, G4 lost, G6 in flight (last, no winner)
                1: hdr(1, 1) + win(1) + hdr(1, 4) + loss(1) + hdr(1, 6),
                # lane 2: G2 lost, G5 timed out (moved past), G7 won
                2: hdr(2, 2) + loss(2) + hdr(2, 5) + hdr(2, 7) + win(2),
                # lane 3: G3 won
                3: hdr(3, 3) + win(3),
            })
            w, l, m, decided = prefix_tally(d, "t")
            # prefix: G1 W, G2 L, G3 W, G4 L, G5 skip, G6 IN FLIGHT -> stop.
            # G7 is decided but waits beyond the in-flight index.
            self.assertEqual((w, l), (2, 2))
            self.assertEqual(m, 5)
            self.assertEqual(decided, 5)

    def test_early_completion_skew_is_excluded(self):
        with TemporaryDirectory() as d:
            # the biased scenario: high indices decided fast, G1 still out
            write_logs(d, {
                1: hdr(1, 1),                       # marathon in flight
                2: hdr(2, 2) + loss(2) + hdr(2, 4) + loss(2),
                3: hdr(3, 3) + loss(3),
            })
            w, l, m, decided = prefix_tally(d, "t")
            self.assertEqual((w, l, m), (0, 0, 0))  # nothing consumable yet
            self.assertEqual(decided, 3)

    def test_all_decided_matches_full_tally(self):
        with TemporaryDirectory() as d:
            write_logs(d, {
                1: hdr(1, 1) + win(1) + hdr(1, 3) + loss(1),
                2: hdr(2, 2) + loss(2),
            })
            w, l, m, decided = prefix_tally(d, "t")
            self.assertEqual((w, l, m, decided), (1, 2, 3, 3))

    def test_empty_and_missing_logs(self):
        with TemporaryDirectory() as d:
            self.assertEqual(prefix_tally(d, "t"), (0, 0, 0, 0))
            write_logs(d, {1: ""})
            self.assertEqual(prefix_tally(d, "t"), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
