"""Negative-evidence item elimination, on one unified constraint layer.

The rule the measurements produced (2026-08-02): an elimination is only
worth encoding where the PRIOR puts real mass on what the observation
refutes. Leftovers is the tier's most common item (16.1%) and 5.0% of our
assumptions about a mon proven Leftovers-less still said Leftovers. The
equivalent Boots case is 0.2%, because the mons that take hazard chip are
the ones whose builds do not run Boots anyway — kept only because it costs
one line once the constraint layer exists.

Eliminations are DEDUCTIVE and safe for every tier to apply blindly;
positive assertions (Boots from zero chip) stay separate because they are
abductive — Magic Guard explains the same observation.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.set_inference import BattleObservations


def obs(events, role="p1"):
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role=role)
    o = BattleObservations()
    o.update(b)
    return o


def test_below_max_hp_through_upkeep_rules_out_leftovers():
    o = obs(["|switch|p2a: Kingambit|Kingambit, M|100/100",
             "|-damage|p2a: Kingambit|60/100",
             "|turn|3"])
    assert "leftovers" in o.forbidden("kingambit")
    assert "blacksludge" in o.forbidden("kingambit")


def test_a_heal_that_turn_clears_the_proof():
    o = obs(["|switch|p2a: Kingambit|Kingambit, M|100/100",
             "|-damage|p2a: Kingambit|60/100",
             "|-heal|p2a: Kingambit|66/100|[from] item: Leftovers",
             "|turn|3"])
    assert "leftovers" not in o.forbidden("kingambit")


def test_full_hp_proves_nothing_about_leftovers():
    o = obs(["|switch|p2a: Kingambit|Kingambit, M|100/100", "|turn|3"])
    assert "leftovers" not in o.forbidden("kingambit")


def test_unstatused_through_upkeep_rules_out_the_orbs():
    o = obs(["|switch|p2a: Ursaluna|Ursaluna, F|100/100",
             "|-damage|p2a: Ursaluna|80/100",
             "|turn|3"])
    assert {"flameorb", "toxicorb"} <= o.forbidden("ursaluna")


def test_a_status_present_clears_the_orb_proof():
    o = obs(["|switch|p2a: Ursaluna|Ursaluna, F|100/100",
             "|-damage|p2a: Ursaluna|80/100",
             "|-status|p2a: Ursaluna|brn|[from] item: Flame Orb",
             "|turn|3"])
    assert "flameorb" not in o.forbidden("ursaluna")


def test_hazard_chip_rules_out_boots():
    o = obs(["|-sidestart|p2: Opp|move: Stealth Rock",
             "|switch|p2a: Corviknight|Corviknight, M|100/100",
             "|-damage|p2a: Corviknight|94/100|[from] Stealth Rock",
             "|turn|5"])
    assert "heavydutyboots" in o.forbidden("corviknight")
    assert "corviknight" not in o.boots      # positive read stays off


def test_our_own_side_is_never_constrained():
    o = obs(["|switch|p1a: Kingambit|Kingambit, M|100/100",
             "|-damage|p1a: Kingambit|60/100",
             "|turn|3"])
    assert not o.forbidden("kingambit")


def test_forbidden_is_empty_for_an_unseen_species():
    assert obs(["|turn|1"]).forbidden("weavile") == frozenset()


def test_a_refuted_item_does_not_discard_the_set(tmp_path):
    """A ruled-out ITEM must not rule out the SET.

    User correction 2026-08-02: most builds tolerate several items that shift
    them subtly — Kingambit's Leftovers vs Black Glasses vs Air Balloon are
    the same set — so dropping the candidate would throw away its moveset and
    spread, the very things the curated tier exists to supply, over one
    refuted axis. Kingambit is the live case: all three curated sets hold
    Leftovers, so a Leftovers-less Kingambit must keep a curated MOVESET and
    have only its item re-drawn.
    """
    import random

    from showdown.gen9_translator import Gen9Translator

    o = obs(["|switch|p2a: Kingambit|Kingambit, M|100/100",
             "|-damage|p2a: Kingambit|60/100",
             "|turn|3"])
    banned = o.forbidden("kingambit")
    assert "leftovers" in banned

    tr = Gen9Translator(set_source="gen9ou")
    tr._obs, tr._rng, tr._prefer_ps = o, random.Random(5), True
    items, movesets = set(), set()
    for _ in range(12):
        s = tr._opp_set("kingambit")
        assert s is not None, "the set must survive a refuted item"
        items.add(s["item"])
        movesets.add(tuple(sorted(s["moves"])))
    assert not (items & banned), f"drew a ruled-out item: {items & banned}"
    assert items - {"leftovers"}, "item must actually be re-drawn"
    # the curated movesets are the point: they must be preserved, not
    # replaced by a chaos composition
    for ms in movesets:
        assert "suckerpunch" in ms and len(ms) == 4, ms


def test_chaos_sampling_respects_the_same_constraint():
    import random
    from showdown.chaos_stats import ChaosStats
    st = ChaosStats(format="gen9ou").pokemon["kingambit"]
    rng = random.Random(4)
    banned = frozenset({"leftovers", "blacksludge"})
    for _ in range(40):
        s = st.sample_set(rng, exclude_items=banned)
        assert s["item"] not in banned


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
