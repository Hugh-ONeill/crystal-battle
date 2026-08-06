"""Every forced-switch exit must name the mon we sent in.

Our own switches emit no protocol event the scanner narrates — the
opponent's "they go to X" has no counterpart on our side — so a replacement
we chose reaches the broadcast ONLY through director.note("we send X in").

The note was wired to two of the three exits: the single-switch shortcut and
the heuristic fallback, on the assumption that the searched branch narrated
itself. It does not, and the searched branch is the one that runs. Measured
over hunts 5 and 6: our mon was knocked out 39 times, we used a pivot 14
times, and "we send X in" reached the record 5 times in total. Not one of
those 5 was a pivot — a pivot almost always leaves several legal
replacements, which is exactly when the single-switch shortcut does not fire.

forceSwitch is read straight off the server request
(poke_env/battle/battle.py: `request.get("forceSwitch", [False])[0]`), and
Showdown sets it for BOTH a knockout replacement and a U-turn, so one branch
covers both and one missing note loses both.
"""
import asyncio
import logging

from poke_env.battle.battle import Battle

from crystal_broadcast.beat_director import Director
from showdown.gen9_player import Gen9PokeEnginePlayer as P
from tests.test_gen9_translator import REQUEST


def _forced_switch_battle():
    b = Battle("battle-gen9ou-test-1", "wizbot", logging.getLogger("test"),
               gen=9)
    b.parse_request(REQUEST)
    b.parse_message(["", "switch", "p1a: Ninetales", "Ninetales, L100, F",
                     "323/323"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M",
                     "100/100"])
    b.parse_message(["", "turn", "1"])
    # the shape a KO replacement and a U-turn both produce
    b.parse_message(["", "faint", "p1a: Ninetales"])
    b.parse_request({
        "forceSwitch": [True],
        "side": {**REQUEST["side"], "pokemon": [
            {**REQUEST["side"]["pokemon"][0], "condition": "0 fnt"},
            *REQUEST["side"]["pokemon"][1:],
        ]},
    })
    assert b.force_switch, "fixture must reach a forced-switch state"
    assert len(b.available_switches) > 1, \
        "several options, so the single-switch shortcut cannot cover it"
    return b


def _player(battle):
    p = P.__new__(P)
    p._airi = object()                 # non-None so the note path is live
    p._director = Director()
    p._last_tag = battle.battle_tag    # skip the per-battle scouting setup
    p._verbose = False
    return p


def _noted(director):
    return [e.prose for e in director._pending
            if (e.prose or "").startswith("we send ")]


def test_the_searched_branch_names_the_replacement():
    """The path that actually runs. Before this, MCTS returned an order and
    the broadcast heard nothing at all."""
    b = _forced_switch_battle()
    p = _player(b)
    picked = b.available_switches[1]
    p._search_samples = lambda battle: []
    p._map_choice = lambda ranked, battle: p.create_order(picked)
    order = asyncio.run(p._choose_move_impl(b))
    assert order is not None
    assert _noted(p._director) == [f"we send {picked.species.title()} in"]


def test_the_heuristic_fallback_still_names_it():
    """Search failing must not also lose the record of what we did next."""
    b = _forced_switch_battle()
    p = _player(b)

    def boom(battle):
        raise RuntimeError("search died")

    p._search_samples = boom
    asyncio.run(p._choose_move_impl(b))
    assert len(_noted(p._director)) == 1


def test_the_note_survives_a_broadcast_that_is_switched_off():
    """_airi None is the bench and ladder configuration; the note must be a
    no-op there rather than an exception in the decision path."""
    b = _forced_switch_battle()
    p = _player(b)
    p._airi = None
    picked = b.available_switches[0]
    p._search_samples = lambda battle: []
    p._map_choice = lambda ranked, battle: p.create_order(picked)
    assert asyncio.run(p._choose_move_impl(b)) is not None
    assert _noted(p._director) == []
