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


def test_the_constraint_can_empty_a_tier_and_that_is_correct(tmp_path):
    """If every curated candidate is refuted, the tier must yield nothing and
    the caller falls through — not return a set the evidence rules out.

    Kingambit is the live case: all three curated sets hold Leftovers, so a
    Kingambit proven Leftovers-less has NO consistent curated build and must
    be drawn from chaos (with the same constraint applied there).
    """
    import showdown.ps_sets as ps_mod
    cls = next(getattr(ps_mod, n) for n in dir(ps_mod)
               if n[0].isupper() and hasattr(getattr(ps_mod, n), "consistent"))
    o = obs(["|switch|p2a: Kingambit|Kingambit, M|100/100",
             "|-damage|p2a: Kingambit|60/100",
             "|turn|3"])
    banned = o.forbidden("kingambit")
    cands = cls().consistent("kingambit")
    assert cands, "fixture assumes curated kingambit sets exist"
    kept = [c for c in cands if c["item"] not in banned]
    assert not kept, "expected every curated kingambit set to be refuted"


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
