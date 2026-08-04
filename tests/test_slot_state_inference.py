"""Substitute HP estimation, Outrage-class rampage locks, and two-turn
charge commitment (2026-08-04, fp cross-audit residuals). Obs level drives
the REAL scanner over crafted protocol; translator level checks the state
the engine actually receives."""
import logging
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poke_env.battle.battle import Battle

from showdown.gen9_translator import Gen9Translator
from showdown.set_inference import BattleObservations

from tests.test_gen9_translator import REQUEST


def _obs(events, role="p1"):
    b = SimpleNamespace(
        _replay_data=[[""] + e.split("|")[1:] for e in events],
        player_role=role)
    o = BattleObservations()
    o.update(b)
    return o


def _battle(*extra):
    b = Battle("battle-gen9monotype-test-2", "wizbot",
               logging.getLogger("test"), gen=9)
    b.parse_request(REQUEST)
    b.parse_message(["", "switch", "p1a: Ninetales", "Ninetales, L100, F",
                     "323/323"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M",
                     "100/100"])
    b.parse_message(["", "turn", "1"])
    for e in extra:
        b.parse_message(e)
    return b


# ---- substitute hit bookkeeping (obs) -----------------------------------

SUB_STORY = [
    "|switch|p2a: Garchomp|Garchomp, M|100/100",
    "|-start|p2a: Garchomp|Substitute",
    "|move|p1a: Ninetales|Flamethrower|p2a: Garchomp",
    "|-activate|p2a: Garchomp|move: Substitute|[damage]",
    "|turn|3",
]


def test_sub_hit_recorded_with_attacker_and_move():
    o = _obs(["|switch|p1a: Ninetales|Ninetales, F|100/100"] + SUB_STORY)
    assert o.opp_sub_hits() == [("ninetales", "flamethrower")]


def test_new_sub_and_broken_sub_clear_the_ledger():
    o = _obs(["|switch|p1a: Ninetales|Ninetales, F|100/100"] + SUB_STORY
             + ["|-start|p2a: Garchomp|Substitute"])
    assert o.opp_sub_hits() == []
    o2 = _obs(["|switch|p1a: Ninetales|Ninetales, F|100/100"] + SUB_STORY
              + ["|-end|p2a: Garchomp|Substitute"])
    assert o2.opp_sub_hits() == []


def test_switch_clears_sub_hits():
    o = _obs(["|switch|p1a: Ninetales|Ninetales, F|100/100"] + SUB_STORY
             + ["|switch|p2a: Kingambit|Kingambit, M|100/100"])
    assert o.opp_sub_hits() == []


# ---- substitute HP estimate (translator) --------------------------------

def test_their_hit_sub_is_estimated_below_full():
    # translate() itself replays the protocol into obs, so the whole chain
    # (activation -> attacker pairing -> engine-calc estimate) runs live
    b = _battle(["", "-start", "p2a: Garchomp", "Substitute"],
                ["", "move", "p1a: Ninetales", "Flamethrower",
                 "p2a: Garchomp"],
                ["", "-activate", "p2a: Garchomp", "move: Substitute",
                 "[damage]"])
    state = Gen9Translator().translate(b)
    sub_max = state.side_two.pokemon[0].maxhp // 4
    assert 1 <= state.side_two.substitute_health < sub_max


def test_their_fresh_sub_stays_at_quarter():
    b = _battle(["", "-start", "p2a: Garchomp", "Substitute"])
    state = Gen9Translator().translate(b)
    assert state.side_two.substitute_health == \
        state.side_two.pokemon[0].maxhp // 4


def test_our_hit_sub_uses_the_binary_read():
    b = _battle(["", "-start", "p1a: Ninetales", "Substitute"],
                ["", "move", "p2a: Garchomp", "Earthquake",
                 "p1a: Ninetales"],
                ["", "-activate", "p1a: Ninetales", "move: Substitute",
                 "[damage]"])
    state = Gen9Translator().translate(b)
    assert state.side_one.substitute_health == 323 // 10


# ---- rampage lock (obs + translator) ------------------------------------

def test_rampage_counts_consecutive_uses_and_ends_on_other_move():
    o = _obs(["|switch|p2a: Garchomp|Garchomp, M|100/100",
              "|move|p2a: Garchomp|Outrage|p1a: X", "|turn|2",
              "|move|p2a: Garchomp|Outrage|p1a: X", "|turn|3"])
    assert o.rampage_for("garchomp") == ("garchomp", "outrage", 2)
    o2 = _obs(["|switch|p2a: Garchomp|Garchomp, M|100/100",
               "|move|p2a: Garchomp|Outrage|p1a: X", "|turn|2",
               "|move|p2a: Garchomp|Earthquake|p1a: X", "|turn|3"])
    assert o2.rampage_for("garchomp") is None


def test_rampage_ends_on_fatigue_cant_and_miss():
    base = ["|switch|p2a: Garchomp|Garchomp, M|100/100",
            "|move|p2a: Garchomp|Outrage|p1a: X"]
    ends = (["|-start|p2a: Garchomp|confusion|[fatigue]"],
            ["|cant|p2a: Garchomp|slp"],
            ["|-miss|p2a: Garchomp|p1a: X"])
    for tail in ends:
        assert _obs(base + tail + ["|turn|2"]).rampage_for("garchomp") is None


def test_rampaging_opponent_is_pinned_to_the_move():
    b = _battle(["", "move", "p2a: Garchomp", "Outrage", "p1a: Ninetales"])
    tr = Gen9Translator()
    state = tr.translate(b)
    chomp = state.side_two.pokemon[0]
    by_id = {m.id: m for m in chomp.moves if m.id != "none"}
    assert by_id["outrage"].disabled is False
    assert all(m.disabled for mid, m in by_id.items() if mid != "outrage")
    assert "lockedmove" in state.side_two.volatile_statuses
    assert state.side_two.volatile_status_durations.lockedmove == 1


# ---- charge commitment (obs + translator) -------------------------------

def test_charge_set_on_prepare_cleared_on_release():
    charge = ["|switch|p2a: Garchomp|Garchomp, M|100/100",
              "|move|p2a: Garchomp|Meteor Beam|p1a: X|[still]",
              "|-prepare|p2a: Garchomp|Meteor Beam"]
    o = _obs(charge)
    assert o.opp_charging() == "meteorbeam"
    o2 = _obs(charge + ["|turn|2",
                        "|move|p2a: Garchomp|Meteor Beam|p1a: X"])
    assert o2.opp_charging() is None


def test_power_herb_release_clears_the_charge():
    o = _obs(["|switch|p2a: Garchomp|Garchomp, M|100/100",
              "|move|p2a: Garchomp|Meteor Beam|p1a: X|[still]",
              "|-prepare|p2a: Garchomp|Meteor Beam",
              "|-enditem|p2a: Garchomp|Power Herb"])
    assert o.opp_charging() is None


def test_charging_opponent_carries_volatile_and_pin():
    b = _battle(["", "move", "p2a: Garchomp", "Solar Beam", "p1a: Ninetales",
                 "[still]"],
                ["", "-prepare", "p2a: Garchomp", "Solar Beam"])
    state = Gen9Translator().translate(b)
    assert "solarbeam" in state.side_two.volatile_statuses
    chomp = state.side_two.pokemon[0]
    by_id = {m.id: m for m in chomp.moves if m.id != "none"}
    assert by_id["solarbeam"].disabled is False
    assert all(m.disabled for mid, m in by_id.items() if mid != "solarbeam")
