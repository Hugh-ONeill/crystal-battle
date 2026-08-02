"""Behavioral Choice disproof: two distinct moves in one stint = not
Choice-locked. Clears wrongly-branded pure attackers that the status-move
veto can't reach (33 assumed-Choice game-mons were falsified by later item
reveals in the 2026-08-02 shadow-corpus measurement), and blocks every tier
(chaos draw, curated candidate, damage/speed inference) from re-branding."""
import random
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.set_inference import BattleObservations
from showdown.chaos_stats import PokemonStats


def _obs(events, role="p1"):
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role=role)
    o = BattleObservations()
    o.update(b)
    return o


SWITCH = "|switch|p2a: Kyurem|Kyurem|100/100"


def test_two_moves_one_stint_disproves_choice():
    o = _obs([SWITCH,
              "|move|p2a: Kyurem|Freeze-Dry|p1a: X",
              "|move|p2a: Kyurem|Earth Power|p1a: X"])
    assert "kyurem" in o.choice_disproven


def test_disproof_clears_sticky_confirmed_choice():
    o = BattleObservations()
    o.confirmed["kyurem"] = "choicespecs"
    b = SimpleNamespace(_replay_data=[[""] + e.split("|")[1:] for e in [
        SWITCH,
        "|move|p2a: Kyurem|Freeze-Dry|p1a: X",
        "|move|p2a: Kyurem|Earth Power|p1a: X"]], player_role="p1")
    o.update(b)
    assert "kyurem" not in o.confirmed


def test_same_move_twice_is_not_disproof():
    o = _obs([SWITCH,
              "|move|p2a: Kyurem|Freeze-Dry|p1a: X",
              "|move|p2a: Kyurem|Freeze-Dry|p1a: X"])
    assert "kyurem" not in o.choice_disproven


def test_moves_split_across_stints_are_not_disproof():
    o = _obs([SWITCH,
              "|move|p2a: Kyurem|Freeze-Dry|p1a: X",
              "|switch|p2a: Ting-Lu|Ting-Lu|100/100",
              SWITCH,
              "|move|p2a: Kyurem|Earth Power|p1a: X"])
    assert "kyurem" not in o.choice_disproven


def test_struggle_and_called_moves_dont_count():
    o = _obs([SWITCH,
              "|move|p2a: Kyurem|Freeze-Dry|p1a: X",
              "|move|p2a: Kyurem|Struggle|p1a: X"])
    assert "kyurem" not in o.choice_disproven
    o = _obs([SWITCH,
              "|move|p2a: Kyurem|Sleep Talk|p2a: Kyurem",
              "|move|p2a: Kyurem|Earth Power|p1a: X|[from]move: Sleep Talk"])
    assert "kyurem" not in o.choice_disproven


def test_our_side_is_not_tracked():
    o = _obs(["|switch|p1a: Kyurem|Kyurem|100/100",
              "|move|p1a: Kyurem|Freeze-Dry|p2a: X",
              "|move|p1a: Kyurem|Earth Power|p2a: X"])
    assert o.choice_disproven == set()


def test_sample_set_exclude_items():
    st = PokemonStats("Testmon", {
        "Raw count": 100, "usage": 0.5,
        "Moves": {"freezedry": 100, "earthpower": 100},
        "Items": {"choicespecs": 90, "leftovers": 10},
        "Abilities": {"pressure": 100},
        "Spreads": {"Timid:0/0/0/252/4/252": 100},
        "Tera Types": {"ice": 100},
    })
    rng = random.Random(1)
    for _ in range(20):
        s = st.sample_set(rng, exclude_items=frozenset(
            {"choiceband", "choicespecs", "choicescarf"}))
        assert s["item"] == "leftovers"


def test_damage_upgrade_gated_by_disproof():
    class _FixedRatio(BattleObservations):
        def _observed_ratio(self, ev, opp_mon, our_mons):
            return 1.5
    o = _FixedRatio()
    o.damage_evidence.append({"species": "kyurem", "move": "freezedry",
                              "damage": 60, "our_species": "corviknight",
                              "weather": "none", "se": False})
    assert o.damage_item_upgrade("kyurem", None, {}) == "choicespecs"
    o.choice_disproven.add("kyurem")
    assert o.damage_item_upgrade("kyurem", None, {}) == "lifeorb"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
