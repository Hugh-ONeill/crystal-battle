"""Regression test for the MCTS empty-options panic.

    pyo3_runtime.PanicException: index out of bounds: the len is 0 but the
    index is 0

killed CBAiri partway through two exhibition games vs FPAiri on 2026-07-19
(gen9ou, suite_v1/03_dnite_tinglu_balance, search-ms 800 and 1500). A
mid-match panic FORFEITS the game, so this is a ladder-deployment bug, not
just a crash.

Mechanism, fixed in poke-engine 565d8c0 (2026-07-26): a forced switch with no
alive reserves fell through a `get_all_options` early return that had no empty
guard, handing MCTS an empty options vec. `Node::expand` indexes
`s1_options[s1_move_index]` unconditionally, and at the ROOT it does not
short-circuit on `battle_is_over`, so the empty list is reached and panics.
The fix restores the "no legal action -> MoveChoice::None" invariant on every
early-return path.

That fix shipped without a test. This pins the invariant at the pyo3 boundary
— i.e. against the built artifact the bot actually runs, which is where the
panic surfaced — rather than in Rust unit-test space.

States are mutated at the serialization level on purpose: the Python
`Pokemon` objects expose read-only attributes, so there is no in-memory way to
faint a side. Field order is from `State::deserialize` (poke-engine
src/state.rs:1000, `hp: split[6]`).
"""

import json
import os
import unittest

import poke_engine as pe

POOL = "showdown/bench/pool_positions.jsonl"
HP_FIELD = 6  # SPECIES,level,t1,t2,bt1,bt2,HP,maxhp,...


def _faint_side_one(state_str: str, faint_bench: bool) -> str:
    """Faint side one's active, and optionally its whole bench.

    Fainting only the active is a normal forced switch (the control). Fainting
    the bench too is the crash case: a forced switch with nothing to switch to.
    """
    sides = state_str.split("/")
    segs = sides[0].split("=")
    mons, tail = segs[:6], segs[6:]
    active = int(tail[0])
    out = []
    for i, mon in enumerate(mons):
        fields = mon.split(",")
        if faint_bench or i == active:
            fields[HP_FIELD] = "0"
        out.append(",".join(fields))
    sides[0] = "=".join(out + tail)
    return "/".join(sides)


@unittest.skipUnless(os.path.exists(POOL), f"{POOL} not present")
class TestEmptyOptionsPanic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(POOL) as fh:
            cls.base = json.loads(fh.readline())["state"]

    def _alive(self, state_str):
        s = pe.State.from_string(state_str)
        return sum(1 for p in s.side_one.pokemon if p.hp > 0)

    def test_mutation_actually_faints(self):
        """Guard the guard: if the serialization layout ever shifts, the
        crash case would silently stop being the crash case and this whole
        file would pass for the wrong reason."""
        self.assertEqual(self._alive(self.base), 6)
        self.assertEqual(self._alive(_faint_side_one(self.base, False)), 5)
        self.assertEqual(self._alive(_faint_side_one(self.base, True)), 0)

    def test_forced_switch_with_live_bench(self):
        state = _faint_side_one(self.base, False)
        result = pe.mcts(pe.State.from_string(state), 200)
        # the five living reserves are the only legal actions
        self.assertEqual(len(result.s1), 5)

    def test_forced_switch_with_no_reserves_does_not_panic(self):
        """The exact crash case, at the two search budgets the live crashes
        used. PanicException derives from BaseException, not Exception, so it
        would escape a bare `except Exception` — assert on the call itself."""
        for ms in (200, 800, 1500):
            with self.subTest(search_ms=ms):
                state = _faint_side_one(self.base, True)
                result = pe.mcts(pe.State.from_string(state), ms)
                # the guard's substitute: exactly one option, MoveChoice::None
                self.assertEqual(len(result.s1), 1)

    def test_both_sides_wiped(self):
        """Both sides out of reserves at once — the two-sided force_switch
        early return, which is a separate branch from the one-sided ones."""
        state = _faint_side_one(self.base, True)
        halves = state.split("/")
        segs = halves[1].split("=")
        halves[1] = "=".join(
            [",".join(f if i != HP_FIELD else "0"
                      for i, f in enumerate(m.split(",")))
             for m in segs[:6]] + segs[6:])
        result = pe.mcts(pe.State.from_string("/".join(halves)), 200)
        self.assertEqual(len(result.s1), 1)
        self.assertEqual(len(result.s2), 1)


if __name__ == "__main__":
    unittest.main()
