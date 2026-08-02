"""Extender items are proven by DURATION, not by revelation.

Heat/Damp/Smooth/Icy Rock, Terrain Extender and Light Clay are all silent —
they emit no protocol event — so waiting for one to be "revealed" never
fired for an opponent and every weather was modelled at 5 turns. Against a
Ninetales running Heat Rock on 94% of its sets that is three turns of sun we
believe we have waited out and have not.

Measured on our own ladder logs (554 weather runs, re-sets counted
separately because Drought re-fires on every switch-in of the setter):
durations are bimodal at 4 and 7 elapsed turns exactly as the mechanic
predicts, and 23% of runs prove an extender.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from showdown.gen9_translator import Gen9Translator


@dataclass(frozen=True)
class Ev:
    name: str


@pytest.fixture(scope="module")
def tr():
    return Gen9Translator(set_source="gen9ou")


def battle(turn, weather=None, fields=None):
    from types import SimpleNamespace as NS
    return NS(turn=turn, weather=weather or {}, fields=fields or {},
              team={}, opponent_team={})


def test_weather_within_five_turns_assumes_no_rock(tr):
    w, turns = tr._weather(battle(3, {Ev("SUNNYDAY"): 1}))
    assert w == pe.Weather.SUN
    assert turns == 3          # 5 - 2 elapsed


def test_weather_surviving_past_five_turns_proves_the_rock(tr):
    """A 5-turn sun cannot still be up 5 turns after it was set."""
    w, turns = tr._weather(battle(7, {Ev("SUNNYDAY"): 1}))
    assert w == pe.Weather.SUN
    assert turns == 2          # 8 - 6 elapsed, not clamped to 1


def test_the_inference_is_not_retroactive(tr):
    """At exactly 5 turns elapsed the weather is observably still up, which
    is already proof — one turn earlier it is not."""
    _, short = tr._weather(battle(5, {Ev("SUNNYDAY"): 1}))   # 4 elapsed
    _, long_ = tr._weather(battle(6, {Ev("SUNNYDAY"): 1}))   # 5 elapsed
    assert short == 1 and long_ == 3


def test_terrain_extender_is_proven_the_same_way(tr):
    t, turns = tr._terrain(battle(7, fields={Ev("GRASSY_TERRAIN"): 1}))
    assert t == pe.Terrain.GRASSY
    assert turns == 2


def test_terrain_within_five_turns_assumes_no_extender(tr):
    t, turns = tr._terrain(battle(3, fields={Ev("PSYCHIC_TERRAIN"): 1}))
    assert t == pe.Terrain.PSYCHIC and turns == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
