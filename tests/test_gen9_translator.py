# Tests for the gen9 monotype translator (poke-env Battle -> poke-engine State).
#
# Battles are constructed by feeding real Showdown protocol shapes through
# poke-env's parser -- the same path a live connection uses -- so these tests
# also pin our assumptions about poke-env's message handling.

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from poke_env.battle.battle import Battle

from showdown.gen9_translator import Gen9Translator, parse_engine_choice


def _req_mon(ident, details, condition, active, stats, moves, ability, item):
    return {
        "ident": ident, "details": details, "condition": condition,
        "active": active, "stats": stats, "moves": moves,
        "baseAbility": ability, "ability": ability, "item": item,
        "pokeball": "pokeball",
    }


# A meta-realistic sun-Fire core (own side: full information via request).
REQUEST = {
    "active": [{"moves": [
        {"move": "Flamethrower", "id": "flamethrower", "pp": 24, "maxpp": 24,
         "target": "normal", "disabled": False},
        {"move": "Solar Beam", "id": "solarbeam", "pp": 16, "maxpp": 16,
         "target": "normal", "disabled": False},
        {"move": "Encore", "id": "encore", "pp": 8, "maxpp": 8,
         "target": "normal", "disabled": False},
        {"move": "Will-O-Wisp", "id": "willowisp", "pp": 24, "maxpp": 24,
         "target": "normal", "disabled": False},
    ]}],
    "side": {"name": "wizbot", "id": "p1", "pokemon": [
        _req_mon("p1: Ninetales", "Ninetales, L100, F", "323/323", True,
                 {"atk": 152, "def": 186, "spa": 240, "spd": 236, "spe": 298},
                 ["flamethrower", "solarbeam", "encore", "willowisp"],
                 "drought", "heatrock"),
        _req_mon("p1: Heatran", "Heatran, L100, M", "386/386", False,
                 {"atk": 194, "def": 248, "spa": 296, "spd": 248, "spe": 253},
                 ["magmastorm", "earthpower", "taunt", "stealthrock"],
                 "flashfire", "airballoon"),
        _req_mon("p1: Volcarona", "Volcarona, L100, M", "391/391", False,
                 {"atk": 140, "def": 166, "spa": 309, "spd": 246, "spe": 299},
                 ["quiverdance", "fierydance", "gigadrain", "morningsun"],
                 "flamebody", "heavydutyboots"),
    ]},
}


def make_battle():
    b = Battle("battle-gen9monotype-test-1", "wizbot",
               logging.getLogger("test"), gen=9)
    b.parse_request(REQUEST)
    b.parse_message(["", "switch", "p1a: Ninetales", "Ninetales, L100, F", "323/323"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M", "100/100"])
    b.parse_message(["", "turn", "1"])
    return b


def _find(side, species):
    for p in side.pokemon:
        if p.id == species:
            return p
    raise AssertionError(f"{species} not found in side")


def test_own_side_exact():
    state = Gen9Translator().translate(make_battle())
    nine = state.side_one.pokemon[0]
    assert nine.id == "ninetales"          # active is slot 0
    assert nine.hp == 323 and nine.maxhp == 323
    assert nine.special_attack == 240 and nine.speed == 298
    assert nine.ability == "drought"
    assert nine.item == "heatrock"
    assert {m.id for m in nine.moves} == {"flamethrower", "solarbeam",
                                          "encore", "willowisp"}
    assert nine.types == ("fire", "typeless")
    assert not nine.terastallized
    heatran = _find(state.side_one, "heatran")
    assert heatran.item == "airballoon" and heatran.attack == 194


def test_side_conditions_boosts_weather():
    b = make_battle()
    b.parse_message(["", "-weather", "SunnyDay"])
    b.parse_message(["", "-sidestart", "p1: wizbot", "Spikes"])
    b.parse_message(["", "-sidestart", "p1: wizbot", "Spikes"])
    b.parse_message(["", "-sidestart", "p2: opponent", "move: Stealth Rock"])
    b.parse_message(["", "-boost", "p2a: Garchomp", "atk", "2"])
    state = Gen9Translator().translate(b)

    assert state.side_one.side_conditions.spikes == 2
    assert state.side_two.side_conditions.stealth_rock == 1
    assert state.side_two.attack_boost == 2
    assert state.side_one.attack_boost == 0
    assert state.weather == "sun"
    # Ninetales' revealed Heat Rock -> 8-turn sun (set this turn: all 8 left)
    assert state.weather_turns_remaining == 8


def test_opp_canonical_fill():
    b = make_battle()
    # a second dragon pins the opponent's monotype
    b.parse_message(["", "switch", "p2a: Latios", "Latios, L100", "100/100"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M", "100/100"])
    b.parse_message(["", "move", "p2a: Garchomp", "Earthquake", "p1a: Ninetales"])
    state = Gen9Translator().translate(b)

    chomp = _find(state.side_two, "garchomp")
    move_ids = [m.id for m in chomp.moves]
    assert "earthquake" in move_ids            # revealed move kept
    assert "none" not in move_ids              # canonical set filled the rest
    assert chomp.ability != "noability"        # canonical ability guessed
    # canonical spread, not the neutral-85 fallback: Garchomp neutral-85 HP
    # is 379; any real spread differs
    assert 300 < chomp.maxhp < 450
    assert chomp.hp == chomp.maxhp             # 100/100 fraction preserved


def test_status_substitute_and_volatiles():
    b = make_battle()
    b.parse_message(["", "-status", "p2a: Garchomp", "brn"])
    b.parse_message(["", "-start", "p2a: Garchomp", "Substitute"])
    b.parse_message(["", "-start", "p1a: Ninetales", "move: Taunt"])
    state = Gen9Translator().translate(b)

    chomp = state.side_two.pokemon[0]
    assert chomp.status == "brn"
    assert "substitute" in state.side_two.volatile_statuses
    assert state.side_two.substitute_health == chomp.maxhp // 4
    assert "taunt" in state.side_one.volatile_statuses
    assert state.side_one.volatile_status_durations.taunt >= 1


def test_unrevealed_slots_are_fainted_dummies():
    state = Gen9Translator().translate(make_battle())
    assert len(state.side_two.pokemon) == 6
    for filler in state.side_two.pokemon[1:]:
        assert filler.hp == 0


def test_state_is_searchable():
    b = make_battle()
    b.parse_message(["", "-weather", "SunnyDay"])
    b.parse_message(["", "-sidestart", "p2: opponent", "move: Stealth Rock"])
    state = Gen9Translator().translate(b)

    # round-trips through the engine's serialization
    rebuilt = pe.State.from_string(state.to_string())
    assert rebuilt.side_one.pokemon[0].id.lower() == "ninetales"

    # and MCTS accepts it end-to-end
    result = pe.monte_carlo_tree_search(state, 50)
    assert result.side_one
    choices = {r.move_choice for r in result.side_one}
    assert any(not c.endswith("-tera") for c in choices)


def test_encore_sets_last_used_move():
    b = make_battle()
    b.parse_message(["", "switch", "p2a: Latios", "Latios, L100", "100/100"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M", "100/100"])
    b.parse_message(["", "move", "p2a: Garchomp", "Earthquake", "p1a: Ninetales"])
    b.parse_message(["", "-start", "p2a: Garchomp", "Encore"])
    state = Gen9Translator().translate(b)

    # engine panics if ENCORE is set without last_used_move pointing at a slot
    assert "encore" in state.side_two.volatile_statuses
    slot = int(state.side_two.last_used_move.split(":")[1])
    assert state.side_two.pokemon[0].moves[slot].id == "earthquake"
    assert pe.monte_carlo_tree_search(state, 20).side_two


def test_encore_without_known_move_is_dropped():
    b = make_battle()
    b.parse_message(["", "-start", "p2a: Garchomp", "Encore"])
    state = Gen9Translator().translate(b)
    assert "encore" not in state.side_two.volatile_statuses
    assert pe.monte_carlo_tree_search(state, 20).side_one


def test_gen9ou_chaos_set_fill():
    b = make_battle()
    b.parse_message(["", "move", "p2a: Garchomp", "Earthquake", "p1a: Ninetales"])
    state = Gen9Translator(set_source="gen9ou").translate(b)

    chomp = _find(state.side_two, "garchomp")
    move_ids = [m.id for m in chomp.moves]
    assert "earthquake" in move_ids            # revealed move kept
    assert "none" not in move_ids              # chaos stats filled the rest
    assert chomp.ability != "noability"
    # un-tera'd opponent carries the chaos-predicted tera type
    assert not chomp.terastallized
    assert chomp.tera_type not in ("", "typeless")


def test_opponent_terastallize_reflected():
    b = make_battle()
    b.parse_message(["", "-terastallize", "p2a: Garchomp", "Fire"])
    state = Gen9Translator(set_source="gen9ou").translate(b)

    chomp = state.side_two.pokemon[0]
    assert chomp.terastallized
    assert chomp.tera_type == "fire"
    # base types stay base (the engine applies tera itself)
    assert chomp.types == ("dragon", "ground")


def test_own_tera_type_mined_from_request():
    # REQUEST carries no teraType key -> fallback is first base type;
    # with teraType present it should be used
    req = {**REQUEST, "side": {**REQUEST["side"], "pokemon": [
        {**REQUEST["side"]["pokemon"][0], "teraType": "Grass"},
        *REQUEST["side"]["pokemon"][1:],
    ]}}
    b = Battle("battle-gen9monotype-test-2", "wizbot",
               logging.getLogger("test"), gen=9)
    b.parse_request(req)
    b.parse_message(["", "switch", "p1a: Ninetales", "Ninetales, L100, F", "323/323"])
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L100, M", "100/100"])
    state = Gen9Translator(set_source="gen9ou").translate(b)
    nine = state.side_one.pokemon[0]
    assert not nine.terastallized
    assert nine.tera_type == "grass"


def test_parse_engine_choice():
    assert parse_engine_choice("switch heatran") == ("switch", "heatran")
    assert parse_engine_choice("flamethrower") == ("move", "flamethrower")
