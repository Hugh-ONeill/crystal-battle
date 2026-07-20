"""Negative-evidence Heavy-Duty Boots (gc-0020): a switch-in over Stealth
Rock that takes zero chip is Boots — unless Magic Guard can explain it.
Drives the REAL BattleObservations scanner over crafted protocol."""
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.set_inference import BattleObservations


def _obs(events, role="p1"):
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role=role)
    o = BattleObservations()
    o.update(b)
    return o


SR = "|-sidestart|p2: Opp|move: Stealth Rock"


def test_zero_chip_over_rocks_infers_boots():
    o = _obs([SR,
              "|switch|p2a: Corviknight|Corviknight, M|100/100",
              "|turn|5"])
    assert o.boots_inferred("corviknight") == "heavydutyboots"


def test_taking_sr_chip_cancels_boots():
    o = _obs([SR,
              "|switch|p2a: Corviknight|Corviknight, M|100/100",
              "|-damage|p2a: Corviknight|94/100|[from] Stealth Rock",
              "|turn|5"])
    assert o.boots_inferred("corviknight") is None
    assert "corviknight" not in o.boots


def test_magic_guard_species_is_ambiguous_not_promoted():
    o = _obs([SR,
              "|switch|p2a: Clefable|Clefable, F|100/100",
              "|turn|5"])
    assert o.boots_inferred("clefable") is None       # search must not guess
    assert "clefable" in o.boots_ambiguous            # but recorded for hedge


def test_no_rocks_no_evidence():
    o = _obs(["|switch|p2a: Corviknight|Corviknight, M|100/100", "|turn|5"])
    assert o.boots_inferred("corviknight") is None


def test_mon_already_in_when_rocks_set_is_not_evidence():
    # Corviknight was active BEFORE we set rocks; it never switched over them
    o = _obs(["|switch|p2a: Corviknight|Corviknight, M|100/100",
              SR,
              "|turn|5"])
    assert o.boots_inferred("corviknight") is None


def test_window_closes_on_next_move_not_just_turn():
    o = _obs([SR,
              "|switch|p2a: Corviknight|Corviknight, M|100/100",
              "|move|p2a: Corviknight|Roost|p2a: Corviknight"])
    assert o.boots_inferred("corviknight") == "heavydutyboots"


def test_final_switch_resolves_at_end_of_batch():
    # replay ends right after the switch (no following turn/move)
    o = _obs([SR, "|switch|p2a: Corviknight|Corviknight, M|100/100"])
    assert o.boots_inferred("corviknight") == "heavydutyboots"


def test_court_change_moves_rocks_to_the_other_side():
    # rocks start on OUR side (p1); Court Change flips them onto p2's side,
    # then their mon switches over them
    o = _obs(["|-sidestart|p1: Us|move: Stealth Rock",
              "|-swapsideconditions|",
              "|switch|p2a: Corviknight|Corviknight, M|100/100",
              "|turn|5"])
    assert o.boots_inferred("corviknight") == "heavydutyboots"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"ok {name}")
    print(f"\n{len(fns)} tests passed")
